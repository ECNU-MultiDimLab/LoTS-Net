import torch.nn as nn
import torch.nn.functional as F

from lotsnet.models.smd import SMD_Module
from lotsnet.models.spatial_stream import TextGuidedCinetwoDeco
from lotsnet.models.queue import SpectralFeatureQueue
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


class LOTSNET_V2_Queue(nn.Module):
    """V2: Has Queue, no Router/Selector, no Text. Enqueues ALL spectral bands."""

    def __init__(self, queue_len=1000, queue_device="cpu", **kwargs):
        super().__init__()

        c_spe = kwargs["c_spe"]
        smd_rank = kwargs["smd_rank"]
        smd_steps = kwargs["smd_steps"]

        self.spec_stage1 = SMD_Module(1, c_spe, 2, smd_rank, smd_steps)
        self.spec_stage2 = SMD_Module(c_spe, c_spe, 2, smd_rank, smd_steps)
        self.spec_stage3 = SMD_Module(c_spe, c_spe, 2, smd_rank, smd_steps)
        self.aux_head = AuxDecoder(in_ch=c_spe, num_classes=kwargs["num_classes"])

        if isinstance(kwargs["img_size"], int):
            img_size = (kwargs["img_size"], kwargs["img_size"])
        else:
            img_size = kwargs["img_size"]

        self.spatial_stream = TextGuidedCinetwoDeco(
            img_size=img_size,
            in_chans=kwargs["in_chans"],
            stem_ch=kwargs["stem_ch"],
            layer_channels=kwargs["layer_channels"],
            text_dim=kwargs["text_dim"],
            attn_dim=kwargs["c_attn"],
            need_text=False,
        )

        h_prime = img_size[0] // 8
        w_prime = img_size[1] // 8
        self.c_spa = kwargs["layer_channels"][-1]

        self.spectral_queue = SpectralFeatureQueue(
            max_length=queue_len,
            c_spe=c_spe,
            c_spa=self.c_spa,
            c_attn=kwargs["c_attn"],
            h_prime=h_prime,
            w_prime=w_prime,
            storage_device=queue_device,
        )

        skip_ch_list = [
            kwargs["stem_ch"],
            kwargs["layer_channels"][0],
            kwargs["layer_channels"][1],
        ]
        self.decoder = DualStreamDecoder(
            c_spa=self.c_spa,
            c_attn=kwargs["c_attn"],
            skip_channels=skip_ch_list,
            num_classes=kwargs["num_classes"],
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

    def forward(self, img, text_spec, text_spa, is_sup=True):
        B, S, H, W = img.shape

        x_spec = img.unsqueeze(1)
        f_spec = self.spec_stage1(x_spec)
        f_spec = self.spec_stage2(f_spec)
        f_spec = self.spec_stage3(f_spec)

        f_all_bands = f_spec.transpose(1, 2)
        f_spec_mean = f_spec.mean(dim=2)
        aux_logits = self.aux_head(f_spec_mean)
        aux_logits = F.interpolate(
            aux_logits, size=(H, W), mode="bilinear", align_corners=False
        )

        if self.training:
            self.spectral_queue.enqueue(f_all_bands)

        f_spa, skips = self.spatial_stream(img, None)
        f_retrieved = self.spectral_queue(f_spa)
        logits = self.decoder(f_spa, f_retrieved, skips)

        if self.training:
            if is_sup:
                return logits, aux_logits
            else:
                return logits, aux_logits, f_spa, f_retrieved
        else:
            return logits, None
