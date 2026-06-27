import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

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

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)


        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)


        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DualStreamDecoder(nn.Module):
    def __init__(
        self,
        c_spa=256,
        c_attn=64,
        skip_channels=[64, 128, 256],
        num_classes=2,
        base_ch=64,
    ):
        super().__init__()


        self.fusion_conv = nn.Sequential(
            nn.Conv2d(
                c_spa + c_attn, base_ch * 4, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
        )


        self.up1 = UpBlock(base_ch * 4, skip_channels[2], base_ch * 2)


        self.up2 = UpBlock(base_ch * 2, skip_channels[1], base_ch)


        self.up3 = UpBlock(base_ch, skip_channels[0], base_ch)


        self.final_conv = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, f_spa, f_retrieved, skips):


        x = torch.cat([f_spa, f_retrieved], dim=1)
        x = self.fusion_conv(x)


        x = self.up1(x, skips[2])
        x = self.up2(x, skips[1])
        x = self.up3(x, skips[0])


        mask = self.final_conv(x)

        return mask


if __name__ == "__main__":
    print("Testing DualStreamDecoder...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    B = 4
    H, W = 256, 256
    C_spa = 64
    C_attn = 64


    skips_shapes = [
        (B, 32, 256, 256),
        (B, 64, 128, 128),
        (B, 128, 64, 64),
    ]
    skip_channels = [s[1] for s in skips_shapes]


    decoder = DualStreamDecoder(
        c_spa=C_spa,
        c_attn=C_attn,
        skip_channels=skip_channels,
        num_classes=2,
        base_ch=32,
    ).to(device)


    f_spa = torch.randn(B, C_spa, 32, 32).to(device)
    f_retrieved = torch.randn(B, C_attn, 32, 32).to(device)
    skips = [torch.randn(*shape).to(device) for shape in skips_shapes]


    output_mask = decoder(f_spa, f_retrieved, skips)

    print(f"Input F_spa: {f_spa.shape}")
    print(f"Input F_retrieved: {f_retrieved.shape}")
    print(f"Output Mask: {output_mask.shape}")


    assert output_mask.shape == (B, 2, H, W), "Output shape mismatch!"
    print(">> Decoder Test Passed!")
