import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SpectralFeatureQueue(nn.Module):

    def __init__(
        self,
        max_length,
        c_spe,
        c_spa,
        c_attn=256,
        h_prime=32,
        w_prime=32,
        storage_device="cpu",
        retrieval_mode="top1",
        chunk_size=32,
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


        self.w_q = nn.Linear(c_spa, c_attn, bias=False)
        self.w_k = nn.Linear(c_spe, c_attn, bias=False)
        self.w_v = nn.Linear(c_spe, c_attn, bias=False)

        self.scale = c_attn ** -0.5


        self.register_buffer(
            "queue_buffer",
            torch.randn(max_length, c_spe, h_prime, w_prime, device=storage_device),
        )
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("is_full", torch.zeros(1, dtype=torch.bool))


    def enqueue(self, f_spec_selected):
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


    def _forward_top1(self, f_spa, queue_data, B, H, W, compute_device):

        q = self.w_q(f_spa.mean(dim=(2, 3)))


        k = self.w_k(queue_data.mean(dim=(2, 3)))


        sim = torch.matmul(q, k.transpose(0, 1)) * self.scale
        top_idx = sim.argmax(dim=-1)


        retrieved = queue_data[top_idx]


        v_flat = rearrange(retrieved, "b c h w -> b (h w) c")
        f_ret = rearrange(self.w_v(v_flat), "b (h w) c -> b c h w", h=H, w=W)
        return f_ret


    def _forward_chunked(self, f_spa, queue_data, B, H, W, compute_device):
        M = queue_data.shape[0]
        HW = H * W

        f_spa_flat = rearrange(f_spa, "b c h w -> b (h w) c")
        q = self.w_q(f_spa_flat)


        running_max = torch.full(
            (B, HW, 1), float("-inf"), device=compute_device, dtype=q.dtype
        )
        running_sum = torch.zeros(B, HW, 1, device=compute_device, dtype=q.dtype)
        running_num = torch.zeros(B, HW, self.c_attn, device=compute_device, dtype=q.dtype)

        for start in range(0, M, self.chunk_size):
            blk = queue_data[start: start + self.chunk_size].to(compute_device)
            m = blk.shape[0]

            kf = rearrange(blk, "m c h w -> (m h w) c")
            k = self.w_k(kf)
            v = self.w_v(kf)

            logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            blk_max = logits.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(running_max, blk_max)

            scale_old = torch.exp(running_max - new_max)
            exp_logits = torch.exp(logits - new_max)

            running_num = running_num * scale_old + torch.matmul(exp_logits, v)
            running_sum = running_sum * scale_old + exp_logits.sum(dim=-1, keepdim=True)
            running_max = new_max

        f_ret_flat = running_num / (running_sum + 1e-8)
        return rearrange(f_ret_flat, "b (h w) c -> b c h w", h=H, w=W)


    def forward(self, f_spa):
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
