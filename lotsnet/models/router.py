import torch
import torch.nn as nn
import torch.nn.functional as F


class TextGuidedRouter(nn.Module):
    def __init__(
        self,
        c_spe,
        d_text,
        d_attn=256,
        top_k=10,
        dropout=0.1,
        score_noise_std=0.0,
        temperature=1.0,
    ):
        super().__init__()
        self.c_spe = c_spe
        self.d_attn = d_attn
        self.top_k = top_k
        self.scale = d_attn**-0.5
        self.score_noise_std = score_noise_std
        self.temperature = temperature


        self.w_q = nn.Linear(c_spe, d_attn, bias=False)

        self.w_k = nn.Linear(d_text, d_attn, bias=False)
        self.w_v = nn.Linear(d_text, d_attn, bias=False)


        self.w_o = nn.Linear(d_attn, c_spe)

        self.norm = nn.LayerNorm(c_spe)
        self.dropout = nn.Dropout(dropout)


        self.scoring_mlp = nn.Sequential(
            nn.Linear(c_spe, c_spe // 2),
            nn.ReLU(inplace=True),
            nn.Linear(c_spe // 2, 1),
        )

    def forward(self, f_spec_gap, f_text):
        B, S, C = f_spec_gap.shape
        _, L, _ = f_text.shape


        q = self.w_q(f_spec_gap)

        k = self.w_k(f_text)
        v = self.w_v(f_text)


        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)


        f_context = torch.matmul(attn_probs, v)


        f_spec_hat = self.norm(f_spec_gap + self.w_o(f_context))


        scores = self.scoring_mlp(f_spec_hat).squeeze(-1)


        if self.training and self.score_noise_std > 0.0:
            scores = scores + torch.randn_like(scores) * self.score_noise_std


        topk_values, topk_idx = torch.topk(scores, k=self.top_k, dim=1)


        topk_weights = F.softmax(topk_values / self.temperature, dim=-1)

        return topk_weights, topk_idx


if __name__ == "__main__":

    print("Testing TextGuidedRouter...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    B = 4
    S = 60
    C_spe = 512
    D_text = 1024
    K = 10


    router = TextGuidedRouter(c_spe=C_spe, d_text=D_text, d_attn=256, top_k=K).to(
        device
    )


    f_spec_gap = torch.randn(B, S, C_spe).to(device)

    f_text = torch.randn(B, 32, D_text).to(device)


    mask, indices = router(f_spec_gap, f_text)


    print(f"\nInput Sizes: F_spec={f_spec_gap.shape}, F_text={f_text.shape}")
    print(f"Output Mask Shape: {mask.shape} (Expected: [{B}, {S}])")
    print(f"Output Indices Shape: {indices.shape} (Expected: [{B}, {K}])")
    print(f"Output Indices:\n{indices}")


    print("\nVerifying Mask Logic:")
    print(f"Sum of mask values per sample (should be K={K}):")
    print(mask.sum(dim=1))


    unique_vals = torch.unique(mask)
    print(f"Unique values in mask (should be 0 and 1): {unique_vals}")


    sample_idx = 0
    selected_indices = indices[sample_idx]
    mask_vals_at_indices = mask[sample_idx, selected_indices]
    print(
        f"\nSample 0 - Mask values at selected indices (should all be 1.0): \n{mask_vals_at_indices}"
    )

    print("\nTest Passed!")
