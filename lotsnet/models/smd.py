import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _get_num_groups(num_channels: int, max_groups: int = 32) -> int:
    """找 num_channels 的最大因子且 <= max_groups，保证 GroupNorm 合法。"""
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


# ==========================================
# Part 1: Core Matrix Decomposition Logic
# ==========================================


class _MatrixDecomposition2DBase(nn.Module):
    def __init__(
        self,
        MD_S=1,
        MD_D=512,
        MD_R=64,
        train_steps=6,
        eval_steps=7,
        inv_t=100,
        eta=0.9,
        rand_init=True,
        update_during_train=False,
    ):
        super().__init__()
        self.S = MD_S  # Stride/grouping factor, usually 1 for standard usage
        self.D = MD_D  # Feature dimension (C_spe)
        self.R = MD_R  # Rank (Low-rank constraint)

        self.train_steps = train_steps
        self.eval_steps = eval_steps

        self.inv_t = inv_t
        self.eta = eta
        self.rand_init = rand_init
        self.update_during_train = update_during_train

    def _build_bases(self, B, S, D, R, cuda=False):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        # Input x shape: (B_total, C, L) -> e.g., (Batch*H*W, C_spe, S_bands)
        B, C, S = x.shape

        # For spectral processing: D=C (features), N=S (sequence/bands)
        # Note: The original code logic uses self.spatial to toggle view.
        # Here we simplify assuming input is already (Batch, Channel, Sequence)
        # We treat Channel as Dimension D, Sequence as N.

        # Adaptation for NMF logic:
        # We need (Batch, D, N)
        # If D is the feature dimension and N is the number of items to decompose.

        D = C
        # N = S

        if not self.rand_init and not hasattr(self, "bases"):
            # 使用 x.is_cuda 而非硬编码 cuda=True，确保与输入设备一致
            bases = self._build_bases(1, self.S, D, self.R, cuda=x.is_cuda)
            self.register_buffer("bases", bases)

        # Bases shape: (B, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=x.is_cuda)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # Compute final coefficient
        coef = self.compute_coef(x, bases, coef)

        # Reconstruct: (B, D, R) @ (B, N, R)^T -> (B, D, N)
        x_recon = torch.bmm(bases, coef.transpose(1, 2))

        if not self.rand_init and (not self.training or self.update_during_train) and not return_bases:
            self.online_update(bases)

        return x_recon

    @torch.no_grad()
    def online_update(self, bases):
        update = bases.mean(dim=0)
        self.bases += self.eta * (update - self.bases)
        self.bases = F.normalize(self.bases, dim=1)


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inv_t = 1  # Override for NMF

    def _build_bases(self, B, S, D, R, cuda=False):
        # S is ignored in simple view, B is total batch
        if cuda:
            bases = torch.rand((B, D, R)).cuda()
        else:
            bases = torch.rand((B, D, R))
        bases = F.normalize(bases, dim=1)
        return bases

    @torch.no_grad()
    def local_step(self, x, bases, coef):
        # Multiplicative Update Rule for NMF
        # x: (B, D, N)
        # bases: (B, D, R)
        # coef: (B, N, R)

        # Update Coef (H)
        # Numerator: x^T @ bases -> (B, N, D) @ (B, D, R) -> (B, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # Denom: coef @ (bases^T @ bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)

        # Update Bases (W)
        # Numerator: x @ coef -> (B, D, N) @ (B, N, R) -> (B, D, R)
        numerator = torch.bmm(x, coef)
        # Denom: bases @ (coef^T @ coef)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # Final projection to get coefficients
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)
        return coef


# ==========================================
# Part 2: Wrapper Modules (Hamburger & SMD)
# ==========================================


class Hamburger(nn.Module):
    def __init__(self, in_channels, ham_type="NMF", md_r=64, train_steps=6,
                 rand_init=True, update_during_train=False):
        super().__init__()

        # GroupNorm 替代 BatchNorm1d：
        # 1. 不维护 running stats，避免梯度检查点重计算时双重更新统计量
        # 2. 不依赖 batch size，适合小 batch 的高光谱场景
        # _get_num_groups 找 in_channels 的最大因子且 <=32，保证整除合法
        self.norm = nn.GroupNorm(
            num_groups=_get_num_groups(in_channels), num_channels=in_channels
        )

        # Lower Bread (Pre-processing)
        if ham_type == "NMF":
            self.lower_bread = nn.Sequential(
                nn.Conv1d(in_channels, in_channels, 1), nn.ReLU(inplace=True)
            )
        else:
            self.lower_bread = nn.Conv1d(in_channels, in_channels, 1)

        # Matrix Decomposition Core
        if ham_type == "NMF":
            self.ham = NMF2D(
                MD_D=in_channels,
                MD_R=md_r,
                train_steps=train_steps,
                rand_init=rand_init,
                update_during_train=update_during_train,
            )
        else:
            raise NotImplementedError(
                f"HAM type {ham_type} not implemented in this simplified module."
            )

    def forward(self, x):
        # x shape: (B_total, C, S) -> e.g. (Batch*H*W, C_spe, Spectral_Bands)
        # 与参考代码一致：residual 保留原始输入（DwConv 输出），不经过 norm
        residual = x

        x = self.norm(x)
        x = self.lower_bread(x)
        x = self.ham(x)

        # relu(X + NMF(Norm(X)))，与论文 Eq.1 及参考代码一致
        out = F.relu(residual + x, inplace=True)
        return out


class SMD_Module(nn.Module):
    """
    Spectral Matrix Decomposition Block
    与参考论文 (MICCAI 2023) 及其源代码对齐的实现：
    Flow: DwConv(DW+PW) -> SMD(Norm+NMF+Residual) -> Norm -> FFN(4x) -> Residual
    对应论文 Eq.1: X=DwConv(Zin), X'=SMD(Norm(X))+X, Zout=FFN(Norm(X'))+X'
    """

    def __init__(
        self,
        in_channels,
        hidden_feature=64,
        spatial_reduction=2,
        md_r=16,
        train_steps=6,
        smd_ema_bases=False,
    ):
        super(SMD_Module, self).__init__()

        # --- 修改开始 ---
        # 拆分为 Depthwise (空间下采样) + Pointwise (通道映射)
        # 1. Depthwise Conv: 负责空间下采样，独立处理每个通道
        # 输入输出通道数必须相同，groups = in_channels
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=spatial_reduction + 1,
            stride=spatial_reduction,
            padding=(spatial_reduction + 1) // 2,
            groups=in_channels,
            bias=False,  # 后接 BN，通常不需要 bias
        )

        # 2. Pointwise Conv: 负责将维度映射到 hidden_feature
        # 1x1 卷积，混合通道信息
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_feature,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        # 2. Low-Rank Spectral Extraction (Hamburger/NMF)
        # smd_ema_bases=True 时：rand_init=False（持久化基矩阵）+训练时也做 EMA 更新
        self.ham = Hamburger(
            hidden_feature,
            ham_type="NMF",
            md_r=md_r,
            train_steps=train_steps,
            rand_init=not smd_ema_bases,
            update_during_train=smd_ema_bases,
        )

        # Norm 2 (Pre-FFN)：GroupNorm 替代 BatchNorm2d，理由同 Hamburger.norm
        self.norm2 = nn.GroupNorm(
            num_groups=_get_num_groups(hidden_feature), num_channels=hidden_feature
        )

        # 3. Spectral FFN
        # 这里的 FFN 是针对 Spectral Channel (C) 的，类似于 Channel Attention 的 MLP
        self.spectral_ffn = nn.Sequential(
            # nn.BatchNorm1d(hidden_feature), # 移到了外部作为 norm2
            nn.Conv1d(hidden_feature, hidden_feature * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_feature * 4, hidden_feature, kernel_size=1),
        )

    def forward(self, x):
        # Input x: (B, C_in, S, H, W)
        b, c_in, s, h, w = x.shape

        # --- Step 1: Spatial Downsampling (DwConv logic) ---
        # 1. Reshape for 2D Conv
        x_spatial = rearrange(x, "b c s h w -> (b s) c h w").contiguous()

        # 2. Depthwise Conv (Spatial Downsampling)
        # Shape: (BS, C_in, H, W) -> (BS, C_in, H', W')
        x_spatial_dw = self.depthwise(x_spatial)

        # 3. Pointwise Conv (Channel Projection)
        # Shape: (BS, C_in, H', W') -> (BS, C_out, H', W')
        x_down = self.pointwise(x_spatial_dw)

        _, c_out, h_new, w_new = x_down.shape

        # --- Step 2: SMD (Matrix Decomposition) ---
        # DwConv 输出直接送入 Hamburger，Hamburger 内部完成 Norm+NMF+Residual
        # 与参考代码一致，不在外部额外加 norm（避免双重归一化）
        x_spectral = rearrange(
            x_down, "(b s) c h w -> (b h w) c s", b=b, s=s
        ).contiguous()

        x_refined = self.ham(x_spectral)

        x_refined_spatial = rearrange(
            x_refined, "(b h w) c s -> (b s) c h w", b=b, h=h_new, w=w_new
        ).contiguous()

        # --- Step 3: FFN ---
        # 以下保持不变 ...
        x_norm2 = self.norm2(x_refined_spatial)

        x_norm2_spectral = rearrange(
            x_norm2, "(b s) c h w -> (b h w) c s", b=b, s=s
        ).contiguous()

        x_ffn = self.spectral_ffn(x_norm2_spectral)

        x_out = x_refined + x_ffn

        # --- Step 4: Restore Dimensions ---
        x_final = rearrange(
            x_out, "(b h w) c s -> b c s h w", b=b, h=h_new, w=w_new
        ).contiguous()

        return x_final


# def forward(self, x):
#     # Input x: (B, C_in, S, H, W)
#     b, c_in, s, h, w = x.shape

#     # --- Step 1: DwConv (Downsample) ---
#     x_spatial = rearrange(x, "b c s h w -> (b s) c h w").contiguous()
#     x_down = self.depthwiseconv(x_spatial)  # -> (B*S, C_out, H', W')

#     # [新增] Apply Norm 1
#     x_norm1 = self.norm1(x_down)

#     _, c_out, h_new, w_new = x_norm1.shape

#     # --- Step 2: SMD (Matrix Decomposition) ---
#     # Rearrange for SMD: (B*S, C, H', W') -> (B*H'*W', C, S)
#     # 注意：这里是将空间像素视为 Batch，通道视为特征，波段视为序列
#     x_spectral = rearrange(
#         x_norm1, "(b s) c h w -> (b h w) c s", b=b, s=s
#     ).contiguous()

#     # Hamburger 内部包含 Residual: out = relu(x + ham(x))
#     # 但标准 Transformer Block 是 x + Attention(Norm(x))
#     # 你的 Hamburger 实现已经包含了 x + ham(x)，但输入没有 Norm (虽然 Hamburger 内部第一层是 Norm)
#     # 这里的逻辑是：x_refined = Ham(x_spectral)
#     x_refined = self.ham(x_spectral)

#     # Reshape back for Norm 2
#     x_refined_spatial = rearrange(
#         x_refined, "(b h w) c s -> (b s) c h w", b=b, h=h_new, w=w_new
#     ).contiguous()

#     # --- Step 3: FFN ---
#     # [新增] Apply Norm 2
#     x_norm2 = self.norm2(x_refined_spatial)

#     # Prepare for FFN: (B*S, C, H', W') -> (B*S*H'*W', C, 1) -> Conv1d
#     # 或者保持 (B*H'*W', C, S) 的形状给 FFN (针对 C 操作)
#     # 你的 FFN 是 Conv1d，期望输入 (N, C, L)。
#     # 这里最自然的理解是 FFN 作用在 Spectral Channel 上。

#     # 让我们复用 x_spectral 的形状逻辑：(B*H'*W', C, S)
#     x_norm2_spectral = rearrange(
#         x_norm2, "(b s) c h w -> (b h w) c s", b=b, s=s
#     ).contiguous()

#     # FFN forward
#     x_ffn = self.spectral_ffn(x_norm2_spectral)

#     # Residual Connection 2 (Post-FFN)
#     x_out = x_refined + x_ffn # 这里加上的是 x_refined (经过 SMD 后的)

#     # --- Step 4: Restore Dimensions ---
#     x_final = rearrange(
#         x_out, "(b h w) c s -> b c s h w", b=b, h=h_new, w=w_new
#     ).contiguous()

#     return x_final

# class SMD_Module(nn.Module):
#     """
#     Spectral Matrix Decomposition Block
#     Integrates: DwConv (Spatial Downsample) -> Hamburger (Low-Rank Spectral) -> FFN
#     """

#     def __init__(
#         self,
#         in_channels,
#         hidden_feature=64,
#         spatial_reduction=2,  # Stride for DwConv (H,W reduction)
#         md_r=16,  # Rank for NMF
#         train_steps=6,
#     ):
#         super(SMD_Module, self).__init__()

#         # 1. Spatial Downsampling & Denoising (DwConv)
#         # Groups=in_channels ensures it's depth-wise (independent per channel)
#         self.depthwiseconv = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=hidden_feature,  # Usually keeps dim or projects
#             kernel_size=spatial_reduction + 1,  # e.g. 3 if stride 2
#             stride=spatial_reduction,
#             padding=(spatial_reduction + 1) // 2,  # Same padding logic
#             groups=1,  # Note: If you want fully depthwise, groups should be in_channels.
#             # But usually we might want to project dimension here too.
#             # If in_channels != hidden_feature, we can't use groups=in_channels easily
#             # without a point-wise conv before.
#             # Let's assume standard Conv2d for downsampling + projection
#             # OR if input dim == output dim, we use groups.
#         )

#         # If we strictly follow "DwConv" and dim changes, we usually do:
#         # Conv2d(in, out, 1) -> DwConv(out, out, k, s, groups=out).
#         # For simplicity in this implementation, we use a standard Strided Conv
#         # to handle both Downsampling and Dimension Projection (LinearProj) simultaneously.
#         # This matches the "Z_in = LinearProj(X)" step combined with DwConv.

#         # 2. Low-Rank Spectral Extraction (Hamburger/NMF)
#         self.ham = Hamburger(
#             hidden_feature, ham_type="NMF", md_r=md_r, train_steps=train_steps
#         )

#         # 3. Spectral FFN
#         self.spectral_ffn = nn.Sequential(
#             nn.BatchNorm1d(hidden_feature),
#             nn.Conv1d(hidden_feature, hidden_feature * 4, kernel_size=1),
#             nn.GELU(),
#             nn.Conv1d(hidden_feature * 4, hidden_feature, kernel_size=1),
#         )

#     def forward(self, x):
#         # Input x: (B, C_in, S, H, W) -> This is the 5D tensor from your logic
#         # But standard PyTorch Conv2d expects 4D.
#         # We treat 'S' (Spectral Bands) as the "Batch" dimension for spatial operations.

#         b, c_in, s, h, w = x.shape

#         # ------------------------------------------------
#         # Step 1: Spatial Downsampling (DwConv logic)
#         # ------------------------------------------------
#         # Reshape to (B*S, C, H, W) to apply 2D Conv spatially
#         x_spatial = rearrange(x, "b c s h w -> (b s) c h w").contiguous()

#         # Apply Conv: Downsample H, W -> H', W'
#         # Also projects channel dimension C_in -> hidden_feature (C_spe)
#         x_down = self.depthwiseconv(x_spatial)  # -> (B*S, C_spe, H', W')

#         # Update dimensions
#         _, c_out, h_new, w_new = x_down.shape

#         # ------------------------------------------------
#         # Step 2: Low-Rank Decomposition (SMD logic)
#         # ------------------------------------------------
#         # We need to treat (H'*W') as the batch size for NMF,
#         # because we want to decompose the Spectral Correlation.
#         # Data format for Hamburger: (N, C, L) where N=Batch, C=Channels, L=Length
#         # Here: N = (B * H' * W'), C = C_spe, L = S

#         # Rearrange: (B*S, C, H', W') -> (B, S, C, H', W') -> (B, H', W', C, S) -> (B*H'*W', C, S)
#         x_spectral = rearrange(
#             x_down, "(b s) c h w -> (b h w) c s", b=b, s=s
#         ).contiguous()

#         # Apply Hamburger (NMF)
#         x_refined = self.ham(x_spectral)

#         # ------------------------------------------------
#         # Step 3: FFN & Residual
#         # ------------------------------------------------
#         # Apply FFN on the spectral channel dimension
#         x_out = self.spectral_ffn(x_refined) + x_refined

#         # ------------------------------------------------
#         # Step 4: Restore Dimensions
#         # ------------------------------------------------
#         # Output: (B, C_spe, S, H', W')
#         x_final = rearrange(
#             x_out, "(b h w) c s -> b c s h w", b=b, h=h_new, w=w_new
#         ).contiguous()

#         return x_final


# 测试代码
if __name__ == "__main__":
    # 模拟输入: Batch=2, Channel=1 (raw), Spectral=60, H=256, W=256
    # 注意：第一次进入时 C_in 通常为 1，随后被投影到 hidden_feature
    input_tensor = torch.randn(2, 1, 60, 256, 256).cuda()

    # 实例化 SMD 模块 (Stage 1)
    # 将 1 维特征投影到 64 维，空间尺寸减半
    smd_layer = SMD_Module(in_channels=1, hidden_feature=64, spatial_reduction=2).cuda()

    output = smd_layer(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    # 预期输出: (2, 64, 60, 128, 128)
