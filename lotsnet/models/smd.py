import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _get_num_groups(num_channels: int, max_groups: int = 32) -> int:
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


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
        self.S = MD_S
        self.D = MD_D
        self.R = MD_R

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

        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):

        B, C, S = x.shape


        D = C


        if not self.rand_init and not hasattr(self, "bases"):

            bases = self._build_bases(1, self.S, D, self.R, cuda=x.is_cuda)
            self.register_buffer("bases", bases)


        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=x.is_cuda)
        else:
            bases = self.bases.repeat(B, 1, 1)

        bases, coef = self.local_inference(x, bases)


        coef = self.compute_coef(x, bases, coef)


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
        self.inv_t = 1

    def _build_bases(self, B, S, D, R, cuda=False):

        if cuda:
            bases = torch.rand((B, D, R)).cuda()
        else:
            bases = torch.rand((B, D, R))
        bases = F.normalize(bases, dim=1)
        return bases

    @torch.no_grad()
    def local_step(self, x, bases, coef):


        numerator = torch.bmm(x.transpose(1, 2), bases)

        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)


        numerator = torch.bmm(x, coef)

        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):

        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)
        return coef


class Hamburger(nn.Module):
    def __init__(self, in_channels, ham_type="NMF", md_r=64, train_steps=6,
                 rand_init=True, update_during_train=False):
        super().__init__()


        self.norm = nn.GroupNorm(
            num_groups=_get_num_groups(in_channels), num_channels=in_channels
        )


        if ham_type == "NMF":
            self.lower_bread = nn.Sequential(
                nn.Conv1d(in_channels, in_channels, 1), nn.ReLU(inplace=True)
            )
        else:
            self.lower_bread = nn.Conv1d(in_channels, in_channels, 1)


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


        residual = x

        x = self.norm(x)
        x = self.lower_bread(x)
        x = self.ham(x)


        out = F.relu(residual + x, inplace=True)
        return out


class SMD_Module(nn.Module):

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


        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=spatial_reduction + 1,
            stride=spatial_reduction,
            padding=(spatial_reduction + 1) // 2,
            groups=in_channels,
            bias=False,
        )


        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_feature,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )


        self.ham = Hamburger(
            hidden_feature,
            ham_type="NMF",
            md_r=md_r,
            train_steps=train_steps,
            rand_init=not smd_ema_bases,
            update_during_train=smd_ema_bases,
        )


        self.norm2 = nn.GroupNorm(
            num_groups=_get_num_groups(hidden_feature), num_channels=hidden_feature
        )


        self.spectral_ffn = nn.Sequential(

            nn.Conv1d(hidden_feature, hidden_feature * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_feature * 4, hidden_feature, kernel_size=1),
        )

    def forward(self, x):

        b, c_in, s, h, w = x.shape


        x_spatial = rearrange(x, "b c s h w -> (b s) c h w").contiguous()


        x_spatial_dw = self.depthwise(x_spatial)


        x_down = self.pointwise(x_spatial_dw)

        _, c_out, h_new, w_new = x_down.shape


        x_spectral = rearrange(
            x_down, "(b s) c h w -> (b h w) c s", b=b, s=s
        ).contiguous()

        x_refined = self.ham(x_spectral)

        x_refined_spatial = rearrange(
            x_refined, "(b h w) c s -> (b s) c h w", b=b, h=h_new, w=w_new
        ).contiguous()


        x_norm2 = self.norm2(x_refined_spatial)

        x_norm2_spectral = rearrange(
            x_norm2, "(b s) c h w -> (b h w) c s", b=b, s=s
        ).contiguous()

        x_ffn = self.spectral_ffn(x_norm2_spectral)

        x_out = x_refined + x_ffn


        x_final = rearrange(
            x_out, "(b h w) c s -> b c s h w", b=b, h=h_new, w=w_new
        ).contiguous()

        return x_final


if __name__ == "__main__":


    input_tensor = torch.randn(2, 1, 60, 256, 256).cuda()


    smd_layer = SMD_Module(in_channels=1, hidden_feature=64, spatial_reduction=2).cuda()

    output = smd_layer(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
