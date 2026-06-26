import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint as grad_ckpt

from lotsnet.models.smd import SMD_Module
from lotsnet.models.router import TextGuidedRouter
from lotsnet.models.selector import SpectralFeatureSelector
from lotsnet.models.queue import SpectralFeatureQueue
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


class LotsNet(nn.Module):
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
        router_top_k=10,
        stem_ch=64,
        layer_channels=[128, 256, 256],
        queue_len=1000,
        queue_device="cpu",
        queue_retrieval_mode="top1",   # "top1" 或 "chunked"
        queue_chunk_size=32,           # chunked 模式下每块的条目数
        router_score_noise_std=0.0,
        router_temperature=1.0,
        smd_ema_bases=False,
    ):
        super().__init__()

        # --- A. Spectral Stream ---
        self.spec_stage1 = SMD_Module(
            in_channels=1,
            hidden_feature=c_spe,
            spatial_reduction=2,
            md_r=smd_rank,
            train_steps=smd_steps,
            smd_ema_bases=smd_ema_bases,
        )
        self.spec_stage2 = SMD_Module(
            in_channels=c_spe,
            hidden_feature=c_spe,
            spatial_reduction=2,
            md_r=smd_rank,
            train_steps=smd_steps,
            smd_ema_bases=smd_ema_bases,
        )
        self.spec_stage3 = SMD_Module(
            in_channels=c_spe,
            hidden_feature=c_spe,
            spatial_reduction=2,
            md_r=smd_rank,
            train_steps=smd_steps,
            smd_ema_bases=smd_ema_bases,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

        # --- B. Router & Selector ---
        self.router = TextGuidedRouter(
            c_spe=c_spe,
            d_text=text_dim,
            d_attn=c_attn,
            top_k=router_top_k,
            score_noise_std=router_score_noise_std,
            temperature=router_temperature,
        )
        self.selector = SpectralFeatureSelector()

        self.aux_head = AuxDecoder(in_ch=c_spe, num_classes=num_classes)

        # --- C. Spatial Stream ---
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.spatial_stream = TextGuidedCinetwoDeco(
            img_size=img_size,
            in_chans=in_chans,
            stem_ch=stem_ch,
            layer_channels=layer_channels,
            text_dim=text_dim,
            attn_dim=c_attn,
        )

        # --- D. Queue ---
        h_prime = img_size[0] // 8
        w_prime = img_size[1] // 8
        self.c_spa = layer_channels[-1]
        self.spectral_queue = SpectralFeatureQueue(
            max_length=queue_len,
            c_spe=c_spe,
            c_spa=self.c_spa,
            c_attn=c_attn,
            h_prime=h_prime,
            w_prime=w_prime,
            storage_device=queue_device,
            retrieval_mode=queue_retrieval_mode,
            chunk_size=queue_chunk_size,
        )

        # --- E. Decoder ---
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

    def forward(self, img, text_spec, text_spa, input_data_type=0):
        """
        input_data_type:
            0 - 有标签数据 (labeled)          ：入队 + 返回 (logits, aux_logits)
            1 - 无标签弱增强 (unlabeled weak)  ：不入队 + 返回 (logits, aux_logits, f_spa, f_retrieved)
            2 - 无标签强增强 (unlabeled strong) ：不入队 + 返回 (logits, aux_logits, f_spa, f_retrieved)
        训练模式下传入其他整数值将直接抛出 ValueError。
        """
        if self.training and input_data_type not in (0, 1, 2):
            raise ValueError(
                f"input_data_type 必须为 0、1 或 2，但收到了 {input_data_type!r}。"
                " (0=有标签, 1=无标签弱增强, 2=无标签强增强)"
            )

        B, S, H, W = img.shape

        # 1. Spectral Stream
        # 训练时对三个 SMD stage 使用 Gradient Checkpointing，显存换时间。
        x_spec = img.unsqueeze(1)
        if self.training:
            f_spec = grad_ckpt(self.spec_stage1, x_spec, use_reentrant=False)
            f_spec = grad_ckpt(self.spec_stage2, f_spec, use_reentrant=False)
            f_spec = grad_ckpt(self.spec_stage3, f_spec, use_reentrant=False)
        else:
            f_spec = self.spec_stage1(x_spec)
            f_spec = self.spec_stage2(f_spec)
            f_spec = self.spec_stage3(f_spec)

        # 2. Router: GAP → Cross-Attention → Scoring → Top-K
        f_spec_perm = rearrange(f_spec, "b c s h w -> b s c h w").contiguous()
        f_spec_gap = (
            self.gap(rearrange(f_spec_perm, "b s c h w -> (b s) c h w"))
            .view(B, S, -1)
            .contiguous()
        )
        topk_weights, indices = self.router(f_spec_gap, text_spec)

        # 3. Selector: 硬选取 Top-K 波段特征图
        f_spec_for_select = f_spec.transpose(1, 2)  # [B, S, C_spe, H', W']
        f_selected = self.selector(f_spec_for_select, indices)  # [B, K, C_spe, H', W']

        # 4. 可微权重加权求和，梯度路径：loss → topk_weights → scoring_mlp / cross-attn
        f_spec_fusion = (f_selected * topk_weights.view(B, -1, 1, 1, 1)).sum(dim=1)

        aux_logits = self.aux_head(f_spec_fusion)
        aux_logits = nn.functional.interpolate(
            aux_logits, size=(H, W), mode="bilinear", align_corners=False
        )

        # 仅有标签数据（input_data_type == 0）允许入队
        if self.training and input_data_type == 0:
            self.spectral_queue.enqueue(f_selected)

        # 5. Spatial Stream（训练时使用 GradCkpt）
        if self.training:
            f_spa, skips = grad_ckpt(
                self.spatial_stream, img, text_spa, use_reentrant=False
            )
        else:
            f_spa, skips = self.spatial_stream(img, text_spa)

        # 6. Queue Retrieval
        f_retrieved = self.spectral_queue(f_spa)

        # 7. Decoder
        logits = self.decoder(f_spa, f_retrieved, skips)
        if self.training:
            if input_data_type == 0:
                return logits, aux_logits
            else:
                return logits, aux_logits, f_spa, f_retrieved
        else:
            return logits, None
