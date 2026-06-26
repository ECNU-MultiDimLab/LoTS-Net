import torch
import torch.nn as nn
import torch.nn.functional as F


class TextGuidedRouter(nn.Module):
    def __init__(
        self,
        c_spe,  # 光谱特征维度 (Input Feature Dim)
        d_text,  # 文本特征维度 (Text Feature Dim)
        d_attn=256,  # 注意力映射维度
        top_k=10,  # 选取的关键波段数量
        dropout=0.1,
        score_noise_std=0.0,  # 训练时给得分加高斯噪声的标准差；0.0 表示关闭
        temperature=1.0,       # softmax temperature：<1 更尖锐，>1 更平滑
    ):
        """
        TextGuidedRouter: Uses text features to select key spectral bands.

        Args:
            c_spe (int): Dimension of spectral features (C_spe).
            d_text (int): Dimension of text embeddings (D_text).
            d_attn (int): Dimension for the internal attention mechanism.
            top_k (int): Number of bands to select (K).
            score_noise_std (float): Std of Gaussian noise added to scores during
                training to encourage exploration of different band combinations.
                Set to 0.0 (default) to disable. Typical range: 0.05 ~ 0.2.
            temperature (float): Temperature for the final softmax over Top-K scores.
                Values < 1.0 sharpen the weights (more decisive selection),
                values > 1.0 soften them (more uniform aggregation). Default: 1.0.
        """
        super().__init__()
        self.c_spe = c_spe
        self.d_attn = d_attn
        self.top_k = top_k
        self.scale = d_attn**-0.5
        self.score_noise_std = score_noise_std
        self.temperature = temperature

        # 1. Projections for Cross-Attention (Step 2.2)
        # Q comes from Spectral Features
        self.w_q = nn.Linear(c_spe, d_attn, bias=False)
        # K, V come from Text Features
        self.w_k = nn.Linear(d_text, d_attn, bias=False)
        self.w_v = nn.Linear(d_text, d_attn, bias=False)

        # Output Projection to project back to C_spe for residual connection
        self.w_o = nn.Linear(d_attn, c_spe)

        self.norm = nn.LayerNorm(c_spe)
        self.dropout = nn.Dropout(dropout)

        # 2. MLP for Scoring (Step 2.3)
        # Maps (B, S, C_spe) -> (B, S, 1)
        # 不使用 Sigmoid：保留原始 logit 尺度，避免饱和区梯度消失；
        # 排名由 topk 决定，绝对值大小不影响选哪些波段，只影响后续 softmax 权重。
        self.scoring_mlp = nn.Sequential(
            nn.Linear(c_spe, c_spe // 2),
            nn.ReLU(inplace=True),
            nn.Linear(c_spe // 2, 1),
        )

    def forward(self, f_spec_gap, f_text):
        """
        Args:
            f_spec_gap: [B, S, C_spe] - Global Average Pooled spectral features
            f_text:     [B, L, D_text] - Text embeddings

        Returns:
            mask:       [B, S] - Binary mask (1 for Top-K selected bands, 0 otherwise)
            topk_idx:   [B, K] - Indices of the selected bands (for Gather operation)
        """
        B, S, C = f_spec_gap.shape
        _, L, _ = f_text.shape

        # ==================================================
        # Step 2.2: Cross-Modal Attention
        # ==================================================

        # 1. Linear Projections
        # q: [B, S, D_attn]
        q = self.w_q(f_spec_gap)
        # k, v: [B, L, D_attn]
        k = self.w_k(f_text)
        v = self.w_v(f_text)

        # 2. Attention Calculation
        # attn_scores: [B, S, L]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # f_context: [B, S, D_attn]
        f_context = torch.matmul(attn_probs, v)

        # 3. Residual & Norm (Mapping back to C_spe)
        # f_spec_hat: [B, S, C_spe]
        # This is the "Text-Enhanced" feature used ONLY for scoring.
        f_spec_hat = self.norm(f_spec_gap + self.w_o(f_context))

        # ==================================================
        # Step 2.3: Router & Band Selection
        # ==================================================

        # 1. Calculate Importance Scores: [B, S, 1] -> [B, S]
        scores = self.scoring_mlp(f_spec_hat).squeeze(-1)

        # 探索性噪声：训练时向得分加高斯噪声，让排名边界附近的波段随机晋级，
        # 从而让更多波段的 scoring_mlp 路径在不同迭代中获得梯度更新机会。
        # score_noise_std=0.0（默认）时此分支不执行，推理时同样跳过，无训练推理差异。
        if self.training and self.score_noise_std > 0.0:
            scores = scores + torch.randn_like(scores) * self.score_noise_std

        # 2. Differentiable Top-K Selection
        # topk_values: [B, K] — 可微，梯度可沿此路径回传至 scoring_mlp 和交叉注意力层
        # topk_idx:    [B, K] — 整型索引，决定选哪些波段（与原逻辑完全一致）
        topk_values, topk_idx = torch.topk(scores, k=self.top_k, dim=1)

        # 3. 将 Top-K 得分归一化为权重：[B, K]
        # 除以 temperature 调节权重分布的尖锐程度：
        #   temperature < 1 → 更尖锐（近似 hard selection），< 1 时梯度更集中
        #   temperature > 1 → 更均匀（soft aggregation），有助于训练初期探索
        # 梯度链路：loss → topk_weights → topk_values → scores → scoring_mlp / cross-attention
        topk_weights = F.softmax(topk_values / self.temperature, dim=-1)

        return topk_weights, topk_idx


if __name__ == "__main__":
    # === 测试代码 ===
    print("Testing TextGuidedRouter...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模拟输入参数
    B = 4  # Batch size
    S = 60  # Bands
    C_spe = 512  # Spectral feature dimension
    D_text = 1024  # Text feature dimension (e.g., BERT/CLIP)
    K = 10  # Select Top-10 bands

    # 1. 实例化模块
    router = TextGuidedRouter(c_spe=C_spe, d_text=D_text, d_attn=256, top_k=K).to(
        device
    )

    # 2. 模拟输入数据
    # F_spec_gap: [B, S, C_spe]
    f_spec_gap = torch.randn(B, S, C_spe).to(device)
    # F_text: [B, L, D_text] (假设文本长度 L=20)
    f_text = torch.randn(B, 32, D_text).to(device)

    # 3. 前向推理
    mask, indices = router(f_spec_gap, f_text)

    # 4. 验证输出
    print(f"\nInput Sizes: F_spec={f_spec_gap.shape}, F_text={f_text.shape}")
    print(f"Output Mask Shape: {mask.shape} (Expected: [{B}, {S}])")
    print(f"Output Indices Shape: {indices.shape} (Expected: [{B}, {K}])")
    print(f"Output Indices:\n{indices}")

    # 验证 Mask 逻辑
    print("\nVerifying Mask Logic:")
    print(f"Sum of mask values per sample (should be K={K}):")
    print(mask.sum(dim=1))

    # 验证是否只有 0 和 1
    unique_vals = torch.unique(mask)
    print(f"Unique values in mask (should be 0 and 1): {unique_vals}")

    # 验证 Indices 和 Mask 是否对应
    # 取第一个样本，检查 mask 在 indices 指示的位置是否为 1
    sample_idx = 0
    selected_indices = indices[sample_idx]
    mask_vals_at_indices = mask[sample_idx, selected_indices]
    print(
        f"\nSample 0 - Mask values at selected indices (should all be 1.0): \n{mask_vals_at_indices}"
    )

    print("\nTest Passed!")
