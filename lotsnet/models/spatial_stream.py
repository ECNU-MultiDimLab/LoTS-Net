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

    def __init__(self, in_ch, out_ch):
        super().__init__()

        mid_ch = out_ch // 4

        self.net = nn.Sequential(
            ConvBnRelu(in_ch, mid_ch, 1),
            ConvBnRelu(mid_ch, mid_ch, 3, padding=1),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + x)


class DownsampleResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid_ch = out_ch // 4

        self.net = nn.Sequential(
            ConvBnRelu(in_ch, mid_ch, 1),
            ConvBnRelu(mid_ch, mid_ch, 3, stride=2, padding=1),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )


        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + self.shortcut(x))


class ResidualBottleneck(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.drb = DownsampleResidualBlock(in_ch, out_ch)


        self.srbs = nn.Sequential(
            *[StandardResidualBlock(out_ch, out_ch) for _ in range(4)]
        )

    def forward(self, x):
        x = self.drb(x)
        x = self.srbs(x)
        return x


class ContextualEncoder(nn.Module):


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
        img_size=256,
        in_chans=60,
        embed_dim=256,
        patch_size=8,
        depth=4,
        num_heads=8,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim


        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=7,
            stride=patch_size,
            padding=3,
        )


        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]


        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, self.grid_size[0], self.grid_size[1])
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)


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

        B, C, H, W = x.shape


        x = self.proj(x)
        H_grid, W_grid = x.shape[2], x.shape[3]


        if (H_grid, W_grid) != (self.pos_embed.shape[2], self.pos_embed.shape[3]):

            pos_embed = F.interpolate(
                self.pos_embed,
                size=(H_grid, W_grid),
                mode="bilinear",
                align_corners=False,
            )
        else:
            pos_embed = self.pos_embed


        x = x + pos_embed


        x = x.flatten(2).transpose(1, 2)


        x = self.blocks(x)
        x = self.norm(x)


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


        i_c = self.proj_c(e_c)
        i_s = self.proj_s(e_s)


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


        i_c = i_c + pos_c
        i_s = i_s + pos_s


        i_c = i_c.flatten(2).transpose(1, 2)
        i_s = i_s.flatten(2).transpose(1, 2)


        i_s_norm = self.norm_s(i_s)
        i_c_norm = self.norm_c(i_c)
        i_prime_c, _ = self.attn_c(query=i_s_norm, key=i_c_norm, value=i_c_norm)
        i_prime_s, _ = self.attn_s(query=i_c_norm, key=i_s_norm, value=i_s_norm)


        feat_c = self.distill_c(i_prime_c)
        feat_s = self.distill_s(i_prime_s)

        concat_feat = torch.cat([feat_c, feat_s], dim=-1)

        if self.output_fuse:
            final_feat = self.fusion_linear(concat_feat)
        else:
            final_feat = concat_feat


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
        need_text=True,
    ):
        super().__init__()
        self.need_text = need_text


        self.encoder_out_ch = layer_channels[-1]
        self.patch_size = 2 ** len(layer_channels)

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.num_patches = (img_size[0] // self.patch_size) * (
            img_size[1] // self.patch_size
        )


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


        if self.need_text:
            self.w_q_s = nn.Linear(self.encoder_out_ch, attn_dim, bias=False)
            self.w_k_s = nn.Linear(text_dim, attn_dim, bias=False)
            self.w_v_s = nn.Linear(text_dim, attn_dim, bias=False)

            self.w_o_s = nn.Linear(attn_dim, self.encoder_out_ch)
            self.norm_s = nn.LayerNorm(self.encoder_out_ch)
            self.scale = attn_dim**-0.5

    def forward(self, x, text_tokens=None):

        e_c, skips = self.context_encoder(x)
        e_s = self.struct_encoder(x)


        fused_feat = self.ciam(e_c, e_s)


        if not self.need_text:

            return fused_feat, skips


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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing TextGuidedCinetwoDeco on {device}")


    input_img = torch.randn(4, 60, 256, 256).to(device)

    input_text = torch.randn(4, 20, 1024).to(device)


    model = TextGuidedCinetwoDeco(
        img_size=256,
        in_chans=60,
        layer_channels=[32, 64, 64],
        text_dim=1024,
        attn_dim=256,
    ).to(device)


    out_feat, out_skips = model(input_img, input_text)

    print(f"Input Image: {input_img.shape}")
    print(f"Input Text:  {input_text.shape}")
    print(f"Output Feature: {out_feat.shape}")


    assert out_feat.shape[1] == 64, "Output channel dimension mismatch"
    assert out_feat.shape[2] == 32, "Output height dimension mismatch"
    print(">> Dimensions Check Passed.")
