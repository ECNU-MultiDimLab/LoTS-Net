import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic Conv-BN-ReLU Block"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsample -> Concat with Skip -> ConvBlock"""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # 1. Upsampling layer (Bilinear usually better for segmentation than TransposeConv)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # 2. Convolution after concatenation
        # Input channels = in_ch (from prev layer) + skip_ch (from encoder)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # 这里的 resize 是为了防止因为奇数尺寸导致的 1 pixel 误差
        # 虽然我们设计是 256->128->... 但为了鲁棒性加上
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DualStreamDecoder(nn.Module):
    def __init__(
        self,
        c_spa=256,  # 空间流输出通道数 (Encoder Output)
        c_attn=64,  # 光谱检索特征通道数 (Queue Retrieval Output)
        skip_channels=[64, 128, 256],  # Skip connections 通道数 (Stem, Layer1, Layer2)
        num_classes=2,  # 分割类别数
        base_ch=64,  # 解码器基础通道数
    ):
        """
        DualStreamDecoder: Fuses spatial and spectral features and decodes to mask.
        Assuming 3 upsampling stages to go from H/8 to H.
        """
        super().__init__()

        # =======================================================
        # Step 4.3 Part A: Fusion (融合)
        # =======================================================
        # F_fused = Conv3x3(Concat(F_spa, F_retrieved))
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(
                c_spa + c_attn, base_ch * 4, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
        )

        # =======================================================
        # Step 4.3 Part B: Decoding (解码)
        # =======================================================
        # 我们假设 Encoder 下采样了 3 次 (H/8)，所以需要上采样 3 次
        # skips 列表通常顺序是 [Stem, Layer1, Layer2]
        # 解码顺序倒着来: Layer2 -> Layer1 -> Stem

        # Stage 1: H/8 -> H/4
        # Input: fused (base*4) + Skip[-1] (usually 256) -> Output: base*2
        self.up1 = UpBlock(base_ch * 4, skip_channels[2], base_ch * 2)

        # Stage 2: H/4 -> H/2
        # Input: base*2 + Skip[-2] (usually 128) -> Output: base
        self.up2 = UpBlock(base_ch * 2, skip_channels[1], base_ch)

        # Stage 3: H/2 -> H
        # Input: base + Skip[-3] (usually 64) -> Output: base
        self.up3 = UpBlock(base_ch, skip_channels[0], base_ch)

        # Final Segmentation Head
        self.final_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, f_spa, f_retrieved, skips):
        """
        Args:
            f_spa:       [B, C_spa, H', W']
            f_retrieved: [B, C_attn, H', W']
            skips:       List of tensors [Skip_Stem, Skip_L1, Skip_L2]
        """
        # 1. Feature Fusion
        # Concat along channel dimension
        x = torch.cat([f_spa, f_retrieved], dim=1)
        x = self.fusion_conv(x)  # -> [B, 256, H', W']

        # 2. Decoding with Skips
        # skips[-1] corresponds to the deepest skip connection (H/4)
        x = self.up1(x, skips[2])  # -> H/4
        x = self.up2(x, skips[1])  # -> H/2
        x = self.up3(x, skips[0])  # -> H

        # 3. Prediction
        mask = self.final_conv(x)  # -> [B, Num_Classes, H, W]

        return mask


# === 测试代码 ===
if __name__ == "__main__":
    print("Testing DualStreamDecoder...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 模拟参数 (根据你之前的 log 和常规设置)
    B = 4
    H, W = 256, 256
    C_spa = 64  # 来自 TextGuidedCinetwoDeco 的输出
    C_attn = 64  # 来自 Queue 的输出

    # 假设 ContextEncoder (ResNet-like) 的 skip 通道
    # Stem(H/2) -> Layer1(H/4) -> Layer2(H/8)
    # 这里的 skips 列表顺序通常对应编码器的层级顺序
    # 假设: Skip0 (Stem) = 32ch, H/1 (如果没下采样) 或 H/2
    # 为了通用性测试，我们模拟常见的 UNet 形状
    # Output is H/8 (32x32)
    skips_shapes = [
        (B, 32, 256, 256),  # Skip 0 (Full Res or H)
        (B, 64, 128, 128),  # Skip 1 (H/2)
        (B, 128, 64, 64),  # Skip 2 (H/4)
    ]
    skip_channels = [s[1] for s in skips_shapes]

    # 2. 实例化模型
    decoder = DualStreamDecoder(
        c_spa=C_spa,
        c_attn=C_attn,
        skip_channels=skip_channels,
        num_classes=2,
        base_ch=32,
    ).to(device)

    # 3. 模拟输入数据
    f_spa = torch.randn(B, C_spa, 32, 32).to(device)
    f_retrieved = torch.randn(B, C_attn, 32, 32).to(device)
    skips = [torch.randn(*shape).to(device) for shape in skips_shapes]

    # 4. 前向传播
    output_mask = decoder(f_spa, f_retrieved, skips)

    print(f"Input F_spa: {f_spa.shape}")
    print(f"Input F_retrieved: {f_retrieved.shape}")
    print(f"Output Mask: {output_mask.shape}")

    # 验证
    assert output_mask.shape == (B, 2, H, W), "Output shape mismatch!"
    print(">> Decoder Test Passed!")
