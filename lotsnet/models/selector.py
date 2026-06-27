import torch
import torch.nn as nn


class SpectralFeatureSelector(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f_spec, indices):
        B, S, C, H, W = f_spec.shape
        B_idx, K = indices.shape

        assert B == B_idx, "Batch size of features and indices must match."


        batch_indices = torch.arange(B, device=f_spec.device).view(B, 1).contiguous()


        f_selected = f_spec[batch_indices, indices]

        return f_selected


if __name__ == "__main__":

    print("Testing SpectralFeatureSelector...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    B = 4
    S = 60
    K = 10
    C = 64
    H, W = 32, 32


    f_spec = torch.zeros(B, S, C, H, W).to(device)
    for s in range(S):
        f_spec[:, s, :, :, :] = s


    indices = torch.randint(0, S, (B, K)).to(device)
    print(f"Sample Indices (Batch 0): {indices[0]}")


    selector = SpectralFeatureSelector()
    f_selected = selector(f_spec, indices)


    print(f"\nInput Shape: {f_spec.shape}")
    print(f"Indices Shape: {indices.shape}")
    print(f"Output Shape: {f_selected.shape}")


    expected_val = indices[0, 0].item()
    actual_val = f_selected[0, 0, 0, 0, 0].item()

    print("\nVerification:")
    print(f"Expected Value for [0,0]: {expected_val}")
    print(f"Actual Value for [0,0]:   {actual_val}")

    if expected_val == actual_val:
        print(">> Value Match: SUCCESS")
    else:
        print(">> Value Match: FAILED")


    is_correct = True
    for b in range(B):
        for k in range(K):
            target_idx = indices[b, k]

            slice_mean = f_selected[b, k].mean().item()
            if not abs(slice_mean - target_idx) < 1e-5:
                is_correct = False
                print(f"Mismatch at Batch {b}, K {k}")

    if is_correct:
        print(">> All batch/channel consistencies verified.")
