import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SpectralFeatureQueue(nn.Module):
    """
    光谱特征容器（FIFO 全局队列），支持两种检索模式：

    retrieval_mode="top1"
        论文 Eq.1 的精确实现：对每个队列条目做全局平均池化得到描述向量，
        计算 [B, M] 相似度后 argmax 选出最匹配的一个条目，再空间逐像素投影输出。
        显存：O(B*M)，与队列大小无关，极低。

    retrieval_mode="chunked"
        Online-softmax 分块密集检索，与原始 Dense Attention 数值完全等价，
        但峰值显存仅为单块大小（chunk_size 个条目），不随 M 线性增长。
        适合需要保留原始密集注意力语义、同时又要降低显存的场景。
    """

    def __init__(
        self,
        max_length,        # 队列最大容量（条目数）
        c_spe,             # 光谱特征通道 (Key/Value 输入维度)
        c_spa,             # 空间特征通道 (Query 输入维度)
        c_attn=256,        # 内部投影维度
        h_prime=32,        # 特征图高度
        w_prime=32,        # 特征图宽度
        storage_device="cpu",      # 缓冲区存放设备 ("cpu" 或 "cuda")
        retrieval_mode="top1",     # "top1" 或 "chunked"
        chunk_size=32,             # chunked 模式下每块的队列条目数
    ):
        super().__init__()

        assert retrieval_mode in ("top1", "chunked"), (
            f"retrieval_mode 必须为 'top1' 或 'chunked'，但收到了 {retrieval_mode!r}"
        )

        self.max_length = max_length
        self.c_spe = c_spe
        self.c_spa = c_spa
        self.c_attn = c_attn
        self.h_prime = h_prime
        self.w_prime = w_prime
        self.storage_device = storage_device
        self.retrieval_mode = retrieval_mode
        self.chunk_size = max(1, chunk_size)

        # 投影层（两种模式共享权重，nn.Linear 对最后一维操作，天然兼容 2D/3D 输入）
        self.w_q = nn.Linear(c_spa, c_attn, bias=False)
        self.w_k = nn.Linear(c_spe, c_attn, bias=False)
        self.w_v = nn.Linear(c_spe, c_attn, bias=False)

        self.scale = c_attn ** -0.5

        # FIFO 缓冲区（register_buffer：随模型保存/加载，但不参与梯度更新）
        self.register_buffer(
            "queue_buffer",
            torch.randn(max_length, c_spe, h_prime, w_prime, device=storage_device),
        )
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("is_full", torch.zeros(1, dtype=torch.bool))

    # ------------------------------------------------------------------
    # 入队
    # ------------------------------------------------------------------
    def enqueue(self, f_spec_selected):
        """
        Args:
            f_spec_selected: [B, K, C_spe, H', W'] — 选中的光谱特征图
        """
        new_features = rearrange(
            f_spec_selected.detach().clone(), "b k c h w -> (b k) c h w"
        ).contiguous().to(self.storage_device)

        n = new_features.shape[0]
        ptr = int(self.ptr)

        if ptr + n <= self.max_length:
            self.queue_buffer[ptr: ptr + n] = new_features
            self.ptr[0] = (ptr + n) % self.max_length
        else:
            tail = self.max_length - ptr
            self.queue_buffer[ptr:] = new_features[:tail]
            self.queue_buffer[: n - tail] = new_features[tail:]
            self.ptr[0] = n - tail
            self.is_full[0] = True

        if self.ptr[0] == 0:
            self.is_full[0] = True

    # ------------------------------------------------------------------
    # 检索：top-1（论文 Eq.1）
    # ------------------------------------------------------------------
    def _forward_top1(self, f_spa, queue_data, B, H, W, compute_device):
        """
        1. 对每个队列条目做全局均值池化 → 描述向量 [M, C_spe]
        2. 对空间特征做均值池化 → 全局描述 [B, C_spa]
        3. 计算 [B, M] 余弦相似度，argmax 选出最匹配条目
        4. 取出该条目 [B, C_spe, H, W]，逐像素投影到 C_attn 输出
        """
        # Query: 全局平均池化后投影 → [B, C_attn]
        q = self.w_q(f_spa.mean(dim=(2, 3)))            # [B, C_attn]

        # Key: 每个条目全局池化后投影 → [M, C_attn]
        k = self.w_k(queue_data.mean(dim=(2, 3)))       # [M, C_attn]

        # 相似度 [B, M]，argmax 得索引
        sim = torch.matmul(q, k.transpose(0, 1)) * self.scale   # [B, M]
        top_idx = sim.argmax(dim=-1)                             # [B]

        # 取最匹配条目：[B, C_spe, H, W]
        retrieved = queue_data[top_idx]

        # 逐像素投影 → [B, C_attn, H, W]
        v_flat = rearrange(retrieved, "b c h w -> b (h w) c")
        f_ret = rearrange(self.w_v(v_flat), "b (h w) c -> b c h w", h=H, w=W)
        return f_ret

    # ------------------------------------------------------------------
    # 检索：分块 Online-Softmax（与原 Dense Attention 数值等价）
    # ------------------------------------------------------------------
    def _forward_chunked(self, f_spa, queue_data, B, H, W, compute_device):
        """
        对队列条目维度 M 做分块，用 online-softmax 累积结果。
        每块只占 O(B * HW * chunk_size * HW) 显存，不生成全量 [B, HW, M*HW] 矩阵。
        """
        M = queue_data.shape[0]
        HW = H * W

        f_spa_flat = rearrange(f_spa, "b c h w -> b (h w) c")  # [B, HW, C_spa]
        q = self.w_q(f_spa_flat)                                 # [B, HW, C_attn]

        # Online-softmax 累加器
        running_max = torch.full(
            (B, HW, 1), float("-inf"), device=compute_device, dtype=q.dtype
        )
        running_sum = torch.zeros(B, HW, 1, device=compute_device, dtype=q.dtype)
        running_num = torch.zeros(B, HW, self.c_attn, device=compute_device, dtype=q.dtype)

        for start in range(0, M, self.chunk_size):
            blk = queue_data[start: start + self.chunk_size].to(compute_device)
            m = blk.shape[0]

            kf = rearrange(blk, "m c h w -> (m h w) c")
            k = self.w_k(kf)            # [m*HW, C_attn]
            v = self.w_v(kf)            # [m*HW, C_attn]

            logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, HW, m*HW]

            blk_max = logits.max(dim=-1, keepdim=True).values           # [B, HW, 1]
            new_max = torch.maximum(running_max, blk_max)

            scale_old = torch.exp(running_max - new_max)
            exp_logits = torch.exp(logits - new_max)                    # [B, HW, m*HW]

            running_num = running_num * scale_old + torch.matmul(exp_logits, v)
            running_sum = running_sum * scale_old + exp_logits.sum(dim=-1, keepdim=True)
            running_max = new_max

        f_ret_flat = running_num / (running_sum + 1e-8)                 # [B, HW, C_attn]
        return rearrange(f_ret_flat, "b (h w) c -> b c h w", h=H, w=W)

    # ------------------------------------------------------------------
    # forward：统一入口，按 retrieval_mode 分发
    # ------------------------------------------------------------------
    def forward(self, f_spa):
        """
        Args:
            f_spa: [B, C_spa, H', W'] — 空间流特征
        Returns:
            f_retrieved: [B, C_attn, H', W']
        """
        B, C, H, W = f_spa.shape
        compute_device = f_spa.device

        curr_size = self.max_length if self.is_full else int(self.ptr)
        if curr_size == 0:
            return torch.zeros(B, self.c_attn, H, W, device=compute_device)

        queue_data = self.queue_buffer[:curr_size].to(compute_device)

        if self.retrieval_mode == "top1":
            return self._forward_top1(f_spa, queue_data, B, H, W, compute_device)
        else:
            return self._forward_chunked(f_spa, queue_data, B, H, W, compute_device)


# === 快速冒烟测试 ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, K, M = 2, 5, 30
    C_spe, C_spa, C_attn = 64, 128, 64
    H, W = 32, 32

    for mode in ("top1", "chunked"):
        q = SpectralFeatureQueue(
            max_length=M, c_spe=C_spe, c_spa=C_spa, c_attn=C_attn,
            h_prime=H, w_prime=W, storage_device="cpu",
            retrieval_mode=mode, chunk_size=8,
        ).to(device)
        q.enqueue(torch.randn(B, K, C_spe, H, W).to(device))
        q.enqueue(torch.randn(B, K, C_spe, H, W).to(device))
        out = q(torch.randn(B, C_spa, H, W).to(device))
        assert out.shape == (B, C_attn, H, W), f"shape mismatch in mode={mode}"
        print(f"[{mode}] output shape: {out.shape}  ✓")
    print("All tests passed.")
