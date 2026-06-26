import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class StandardResidualBlock(nn.Module):
    """
    SRB: 标准残差块 (Fig. 2c)
    结构: 1x1 -> 3x3 (stride=1) -> 1x1, 尺寸不变
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 为了遵循 bottleneck 设计 (降维->卷积->升维)
        mid_ch = out_ch // 4

        self.net = nn.Sequential(
            ConvBnRelu(in_ch, mid_ch, 1),
            ConvBnRelu(mid_ch, mid_ch, 3, padding=1),  # stride=1
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + x)


class DownsampleResidualBlock(nn.Module):
    """
    DRB: 下采样残差块 (Fig. 2b)
    结构: 1x1 -> 3x3 (stride=2) -> 1x1, 尺寸减半
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid_ch = out_ch // 4

        self.net = nn.Sequential(
            ConvBnRelu(in_ch, mid_ch, 1),
            ConvBnRelu(mid_ch, mid_ch, 3, stride=2, padding=1),  # 下采样发生在这里
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        # Shortcut 路径需要匹配维度和尺寸
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=2, bias=False),  # Downsample
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + self.shortcut(x))


class ResidualBottleneck(nn.Module):
    """
    RB: 论文公式 (1) RB(x) = 4SRB(DRB(x))
    包含 1 个 DRB 和 4 个 SRB
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 1. DRB: 执行下采样和通道数变更 (in_ch -> out_ch)
        self.drb = DownsampleResidualBlock(in_ch, out_ch)

        # 2. 4 x SRB: 保持尺寸和通道数 (out_ch -> out_ch)
        self.srbs = nn.Sequential(
            *[StandardResidualBlock(out_ch, out_ch) for _ in range(4)]
        )

    def forward(self, x):
        x = self.drb(x)
        x = self.srbs(x)
        return x


class ContextualEncoder(nn.Module):
    # CNN 天然支持任意尺寸，只需确保 forward 返回的特征尺寸正确
    # 初始化不需要变，forward 也不需要变，因为它全是卷积
    # 这里直接复用原代码即可
    def __init__(self, in_chans=60, stem_ch=64, layer_channels=[128, 256, 256]):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_chans, stem_ch, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.ModuleList()
        current_in_ch = stem_ch
        for ch in layer_channels:
            rb = ResidualBottleneck(in_ch=current_in_ch, out_ch=ch)
            self.layers.append(rb)
            current_in_ch = ch

    def forward(self, x):
        features = []
        x = self.stem(x)
        features.append(x)
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return x, features


class StructuralEncoder(nn.Module):
    def __init__(
        self,
        img_size=256,  # 这里的 img_size 仅用于初始化 PosEmbed 的默认大小
        in_chans=60,
        embed_dim=256,
        patch_size=8,
        depth=4,
        num_heads=8,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 1. Patch Embedding (Conv2d)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=7,
            stride=patch_size,
            padding=3,
        )

        # 2. Position Embedding
        # 初始化一个默认尺寸的 PosEmbed (例如针对 256x256)
        # 如果输入尺寸变了，我们会动态插值
        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # PosEmbed 形状: [1, Embed_Dim, H_grid, W_grid]
        # 注意：为了方便插值，我们先保存为 2D 形式，而不是 Flatten 后的 1D
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, self.grid_size[0], self.grid_size[1])
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # 1. Patch Embedding
        # x: (B, D, H_grid, W_grid)
        x = self.proj(x)
        H_grid, W_grid = x.shape[2], x.shape[3]

        # 2. Add Position Embedding (动态插值)
        # 检查当前特征图尺寸是否与 PosEmbed 一致
        if (H_grid, W_grid) != (self.pos_embed.shape[2], self.pos_embed.shape[3]):
            # 使用双线性插值调整 PosEmbed 大小
            pos_embed = F.interpolate(
                self.pos_embed,
                size=(H_grid, W_grid),
                mode="bilinear",
                align_corners=False,
            )
        else:
            pos_embed = self.pos_embed

        # 加位置编码 (广播机制)
        x = x + pos_embed

        # 3. Flatten for Transformer
        # (B, D, H_grid, W_grid) -> (B, D, N) -> (B, N, D)
        x = x.flatten(2).transpose(1, 2)

        # 4. Transformer Blocks
        x = self.blocks(x)
        x = self.norm(x)

        # 5. Reshape back
        # (B, N, D) -> (B, D, N) -> (B, D, H_grid, W_grid)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H_grid, W_grid)

        return x


class CrossIndicationAttentionModule(nn.Module):
    def __init__(
        self, in_dim=256, num_patches=1024, num_heads=8, dropout=0.1, output_fuse=True
    ):
        super().__init__()
        self.in_dim = in_dim
        self.output_fuse = output_fuse

        self.proj_c = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.proj_s = nn.Conv2d(in_dim, in_dim, kernel_size=1)

        # Position Embedding: 同样改为 [1, C, H, W] 格式以便插值
        # num_patches 此时仅用于推断默认 H,W (假设正方形)
        grid_side = int(math.sqrt(num_patches))
        self.pos_embed_c = nn.Parameter(torch.zeros(1, in_dim, grid_side, grid_side))
        self.pos_embed_s = nn.Parameter(torch.zeros(1, in_dim, grid_side, grid_side))

        self.norm_c = nn.LayerNorm(in_dim)
        self.attn_c = nn.MultiheadAttention(
            in_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_s = nn.LayerNorm(in_dim)
        self.attn_s = nn.MultiheadAttention(
            in_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.distill_c = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.distill_s = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.GELU(), nn.Dropout(dropout)
        )

        if self.output_fuse:
            self.fusion_linear = nn.Linear(in_dim * 2, in_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed_c, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_s, std=0.02)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, e_c, e_s):
        B_C, C_C, H_C, W_C = e_c.shape
        B_S, C_S, H_S, W_S = e_s.shape
        # print("CIAM Input feature shapes:", e_c.shape, e_s.shape)

        # 1. Proj
        i_c = self.proj_c(e_c)
        i_s = self.proj_s(e_s)

        # 2. Dynamic Pos Embed
        if (H_C, W_C) != (self.pos_embed_c.shape[2], self.pos_embed_c.shape[3]):
            pos_c = nn.functional.interpolate(
                self.pos_embed_c, size=(H_C, W_C), mode="bilinear", align_corners=False
            )
        else:
            pos_c = self.pos_embed_c

        if (H_S, W_S) != (self.pos_embed_s.shape[2], self.pos_embed_s.shape[3]):
            pos_s = nn.functional.interpolate(
                self.pos_embed_s, size=(H_S, W_S), mode="bilinear", align_corners=False
            )
        else:
            pos_s = self.pos_embed_s

        # print("CIAM Projected feature shapes:", i_c.shape, i_s.shape)
        # print(f"CIAM PosEmbed shapes: pos_c {pos_c.shape}, pos_s {pos_s.shape}")
        i_c = i_c + pos_c
        i_s = i_s + pos_s

        # 3. Flatten
        i_c = i_c.flatten(2).transpose(1, 2)  # [B, N, C]
        i_s = i_s.flatten(2).transpose(1, 2)

        # 4. Attention (不变)
        i_s_norm = self.norm_s(i_s)
        i_c_norm = self.norm_c(i_c)
        i_prime_c, _ = self.attn_c(query=i_s_norm, key=i_c_norm, value=i_c_norm)
        i_prime_s, _ = self.attn_s(query=i_c_norm, key=i_s_norm, value=i_s_norm)

        # 5. Fusion (不变)
        feat_c = self.distill_c(i_prime_c)
        feat_s = self.distill_s(i_prime_s)
        # print("CIAM Distilled feature shapes:", feat_c.shape, feat_s.shape)
        concat_feat = torch.cat([feat_c, feat_s], dim=-1)

        if self.output_fuse:
            final_feat = self.fusion_linear(concat_feat)
        else:
            final_feat = concat_feat

        # 6. Reshape
        # 此时 N = H * W，直接 reshape 回 (B, C, H, W)
        out = final_feat.transpose(1, 2).reshape(B_C, -1, H_C, W_C)
        return out


class TextGuidedCinetwoDeco(nn.Module):
    def __init__(
        self,
        img_size=256,
        in_chans=60,
        stem_ch=64,
        layer_channels=[128, 256, 256],
        ciam_heads=8,
        ciam_dropout=0.1,
        ciam_output_fuse=True,
        text_dim=1024,
        attn_dim=256,
        need_text=True,  # [新增] 控制是否启用文本交互
    ):
        """
        TextGuidedCinetwoDeco: Spatial Stream with Optional Text Semantic Injection.
        """
        super().__init__()
        self.need_text = need_text  # 记录标志位

        # 1. 基础参数
        self.encoder_out_ch = layer_channels[-1]
        self.patch_size = 2 ** len(layer_channels)

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.num_patches = (img_size[0] // self.patch_size) * (
            img_size[1] // self.patch_size
        )

        # 2. 原始 CINET 组件 (无论是否有文本都需要)
        self.context_encoder = ContextualEncoder(
            in_chans=in_chans, stem_ch=stem_ch, layer_channels=layer_channels
        )

        self.struct_encoder = StructuralEncoder(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=self.encoder_out_ch,
            patch_size=self.patch_size,
            depth=4,
            num_heads=8,
        )

        self.ciam = CrossIndicationAttentionModule(
            in_dim=self.encoder_out_ch,
            num_patches=self.num_patches,
            num_heads=ciam_heads,
            dropout=ciam_dropout,
            output_fuse=ciam_output_fuse,
        )

        # 3. [条件加载] 空间-文本语义注入模块
        if self.need_text:
            self.w_q_s = nn.Linear(self.encoder_out_ch, attn_dim, bias=False)
            self.w_k_s = nn.Linear(text_dim, attn_dim, bias=False)
            self.w_v_s = nn.Linear(text_dim, attn_dim, bias=False)

            self.w_o_s = nn.Linear(attn_dim, self.encoder_out_ch)
            self.norm_s = nn.LayerNorm(self.encoder_out_ch)
            self.scale = attn_dim**-0.5

    def forward(self, x, text_tokens=None):
        """
        Args:
            x: [B, C_in, H, W]
            text_tokens: [B, L, D_text] (仅当 need_text=True 时需要)
        """
        # 1. Encode
        e_c, skips = self.context_encoder(x)
        e_s = self.struct_encoder(x)

        # 2. CIAM Fusion
        fused_feat = self.ciam(e_c, e_s)

        # 3. [条件执行] 文本交互
        if not self.need_text:
            # 如果不需要文本，直接返回 CIAM 的结果
            return fused_feat, skips

        # 以下是 need_text=True 时的逻辑
        B, C, H_prime, W_prime = fused_feat.shape
        x_flat = rearrange(fused_feat, "b c h w -> b (h w) c").contiguous()

        q = self.w_q_s(x_flat)
        k = self.w_k_s(text_tokens)
        v = self.w_v_s(text_tokens)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        x_context = torch.matmul(attn_probs, v)

        x_enhanced = self.norm_s(x_flat + self.w_o_s(x_context))

        fused_feat_enhanced = rearrange(
            x_enhanced, "b (h w) c -> b c h w", h=H_prime, w=W_prime
        ).contiguous()

        return fused_feat_enhanced, skips


# === 测试代码 ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing TextGuidedCinetwoDeco on {device}")

    # 1. 模拟输入
    # HSI Image
    input_img = torch.randn(4, 60, 256, 256).to(device)
    # Text Tokens (假设来自 BioBERT Large, dim=1024, seq_len=20)
    input_text = torch.randn(4, 20, 1024).to(device)

    # 2. 实例化模型
    # C_spa (encoder_out_ch) 默认为 256
    model = TextGuidedCinetwoDeco(
        img_size=256,
        in_chans=60,
        layer_channels=[32, 64, 64],
        text_dim=1024,
        attn_dim=256,
    ).to(device)

    # 3. 前向传播
    out_feat, out_skips = model(input_img, input_text)

    print(f"Input Image: {input_img.shape}")
    print(f"Input Text:  {input_text.shape}")
    print(f"Output Feature: {out_feat.shape}")
    # 预期: [2, 256, 32, 32] (256/8 = 32)

    # 验证维度一致性
    assert out_feat.shape[1] == 64, "Output channel dimension mismatch"
    assert out_feat.shape[2] == 32, "Output height dimension mismatch"
    print(">> Dimensions Check Passed.")


# import torch
# import torch.nn as nn
# from models.CINET.encoders import ContextualEncoder, StructuralEncoder
# from models.CINET.ciam import CrossIndicationAttentionModule


# class CINET_WO_DECO(nn.Module):
#     def __init__(
#         self,
#         img_size=256,  # 默认参考尺寸，用于初始化位置编码
#         in_chans=60,  # 对应 HSI 的波段数
#         stem_ch=64,
#         layer_channels=[128, 256, 256],  # 3层结构，最终下采样率通常为 2^3=8
#         ciam_heads=8,
#         ciam_dropout=0.1,
#         ciam_output_fuse=True,
#     ):
#         """
#         CINET feature extractor without PixelDecoder.
#         Output: Fused Features (from CIAM) and Skip Connections (from Contextual Encoder).
#         """
#         super().__init__()

#         # 1. 参数设置
#         self.encoder_out_ch = layer_channels[-1]
#         self.patch_size = 2 ** len(layer_channels)  # 通常为 8

#         # 计算参考用的 num_patches (仅用于初始化 PosEmbed 权重大小)
#         if isinstance(img_size, int):
#             img_size = (img_size, img_size)
#         self.num_patches = (img_size[0] // self.patch_size) * (
#             img_size[1] // self.patch_size
#         )

#         # 2. 模块实例化
#         # 2.1 上下文编码器 (CNN Branch)
#         self.context_encoder = ContextualEncoder(
#             in_chans=in_chans, stem_ch=stem_ch, layer_channels=layer_channels
#         )

#         # 2.2 结构编码器 (Transformer Branch)
#         self.struct_encoder = StructuralEncoder(
#             img_size=img_size,
#             in_chans=in_chans,
#             embed_dim=self.encoder_out_ch,
#             patch_size=self.patch_size,
#             depth=4,
#             num_heads=8,
#         )

#         # 2.3 交叉指示注意力模块 (CIAM)
#         self.ciam = CrossIndicationAttentionModule(
#             in_dim=self.encoder_out_ch,
#             num_patches=self.num_patches,
#             num_heads=ciam_heads,
#             dropout=ciam_dropout,
#             output_fuse=ciam_output_fuse,
#         )

#     def forward(self, x):
#         """
#         Args:
#             x: [B, C, H, W] - HSI Input
#         Returns:
#             fused_feat: [B, C_out, H/8, W/8] - 也就是 F_spa
#             skips: List of tensors - 跳跃连接特征，用于后续 Decoder
#         """
#         # 1. Encode - Contextual Stream (CNN)
#         # e_c: [B, C_out, H/8, W/8]
#         # skips: List of [B, C_i, H_i, W_i]
#         e_c, skips = self.context_encoder(x)

#         # 2. Encode - Structural Stream (Transformer)
#         # e_s: [B, C_out, H/8, W/8]
#         e_s = self.struct_encoder(x)

#         # 3. Interact - CIAM Fusion
#         # fused_feat: [B, C_out, H/8, W/8]
#         fused_feat = self.ciam(e_c, e_s)

#         return fused_feat, skips


# # === 测试代码 ===
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Testing CINET_WO_DECO (Feature Extractor Only) on {device}")

#     # 模拟 HSI 输入: Batch=2, Bands=60, H=256, W=256
#     # 注意：输入尺寸必须能被 patch_size (8) 整除
#     input_tensor = torch.randn(4, 60, 256, 256).to(device)

#     # 实例化模型
#     model = CINET_WO_DECO(img_size=256, in_chans=60, layer_channels=[128, 256, 512]).to(
#         device
#     )

#     # 前向传播
#     f_spa, skips = model(input_tensor)

#     print(f"Input Shape: {input_tensor.shape}")
#     print(f"Output Feature (F_spa) Shape: {f_spa.shape}")
#     # 预期: [2, 256, 32, 32] (如果 layer_channels[-1]=256 且下采样8倍)

#     print("Skip Connections Shapes:")
#     for i, s in enumerate(skips):
#         print(f"  Skip {i}: {s.shape}")
