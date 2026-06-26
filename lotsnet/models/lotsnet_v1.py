import torch.nn as nn
from lotsnet.models.smd import SMD_Module
from lotsnet.models.spatial_stream import TextGuidedCinetwoDeco
from lotsnet.models.decoder import DualStreamDecoder


class AuxDecoder(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, num_classes, 1),
        )

    def forward(self, x):
        return self.conv(x)


class LOTSNET_V1_Base(nn.Module):
    """V1: Spectral-only baseline (no Router, no Queue, no Text)."""

    def __init__(
        self,
        img_size=(256, 256),
        in_chans=60,
        num_classes=2,
        text_dim=1024,
        c_spe=64,
        c_attn=64,
        smd_rank=16,
        smd_steps=6,
        stem_ch=64,
        layer_channels=[128, 256, 256],
        **kwargs,
    ):
        super().__init__()

        self.spec_stage1 = SMD_Module(1, c_spe, 2, smd_rank, smd_steps)
        self.spec_stage2 = SMD_Module(c_spe, c_spe, 2, smd_rank, smd_steps)
        self.spec_stage3 = SMD_Module(c_spe, c_spe, 2, smd_rank, smd_steps)

        self.spec_proj = nn.Conv2d(c_spe, c_attn, kernel_size=1)
        self.aux_head = AuxDecoder(in_ch=c_spe, num_classes=num_classes)

        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        self.spatial_stream = TextGuidedCinetwoDeco(
            img_size=img_size,
            in_chans=in_chans,
            stem_ch=stem_ch,
            layer_channels=layer_channels,
            text_dim=text_dim,
            attn_dim=c_attn,
            need_text=False,
        )

        self.c_spa = layer_channels[-1]
        skip_ch_list = [stem_ch, layer_channels[0], layer_channels[1]]

        self.decoder = DualStreamDecoder(
            c_spa=self.c_spa,
            c_attn=c_attn,
            skip_channels=skip_ch_list,
            num_classes=num_classes,
            base_ch=64,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, img, text_spec=None, text_spa=None, is_sup=True):
        B, S, H, W = img.shape

        x_spec = img.unsqueeze(1)
        f_spec = self.spec_stage1(x_spec)
        f_spec = self.spec_stage2(f_spec)
        f_spec = self.spec_stage3(f_spec)

        f_spec_mean = f_spec.mean(dim=2)
        aux_logits = self.aux_head(f_spec_mean)
        aux_logits = nn.functional.interpolate(
            aux_logits, size=(H, W), mode="bilinear", align_corners=False
        )

        f_spec_interaction = self.spec_proj(f_spec_mean)

        f_spa, skips = self.spatial_stream(img, None)
        logits = self.decoder(f_spa, f_spec_interaction, skips)

        if self.training:
            if is_sup:
                return logits, aux_logits
            else:
                return logits, aux_logits, f_spa, None
        else:
            return logits, None
