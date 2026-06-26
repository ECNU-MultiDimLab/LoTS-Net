import torch
import torch.nn as nn


class SpectralFeatureSelector(nn.Module):
    def __init__(self):
        """
        SpectralFeatureSelector: Selects specific spectral bands based on indices.
        No learnable parameters.
        """
        super().__init__()

    def forward(self, f_spec, indices):
        """
        Args:
            f_spec:  [B, S, C, H, W] - The full spectral feature map.
            indices: [B, K]          - The indices of bands to select.

        Returns:
            f_selected: [B, K, C, H, W] - The selected feature maps.
        """
        B, S, C, H, W = f_spec.shape
        B_idx, K = indices.shape

        assert B == B_idx, "Batch size of features and indices must match."

        # =======================================================
        # Advanced Indexing Logic
        # =======================================================
        # We want to select bands for each batch independently.
        # f_spec[b, indices[b, k], :, :, :]

        # 1. Create Batch Indices: [0, 1, ..., B-1] -> [B, 1]
        batch_indices = torch.arange(B, device=f_spec.device).view(B, 1).contiguous()

        # 2. Use Advanced Indexing
        # PyTorch allows indexing with [Batch_Indices, Selection_Indices]
        # batch_indices broadcasts to [B, K] against indices
        # Result shape: [B, K, C, H, W]
        f_selected = f_spec[batch_indices, indices]

        return f_selected


if __name__ == "__main__":
    # === 测试代码 ===
    print("Testing SpectralFeatureSelector...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 模拟输入参数
    B = 4
    S = 60  # 原始波段数
    K = 10  # 选出的波段数
    C = 64  # C_spe
    H, W = 32, 32  # H/8, W/8 (假设原图256)

    # 2. 模拟数据
    # 为了验证选择的正确性，我们让 f_spec 的值具有唯一标识性
    # 比如第 b 个样本，第 s 个波段，全填 s
    f_spec = torch.zeros(B, S, C, H, W).to(device)
    for s in range(S):
        f_spec[:, s, :, :, :] = s

    # 模拟索引 (Router 的输出)
    # 随机生成 [0, 59] 之间的索引
    indices = torch.randint(0, S, (B, K)).to(device)
    print(f"Sample Indices (Batch 0): {indices[0]}")

    # 3. 实例化与推理
    selector = SpectralFeatureSelector()
    f_selected = selector(f_spec, indices)

    # 4. 验证维度
    print(f"\nInput Shape: {f_spec.shape}")
    print(f"Indices Shape: {indices.shape}")
    print(f"Output Shape: {f_selected.shape}")  # 预期: [4, 10, 64, 32, 32]

    # 5. 验证数值正确性
    # 检查 Batch 0 的第 0 个选择是否对应
    expected_val = indices[0, 0].item()
    actual_val = f_selected[0, 0, 0, 0, 0].item()

    print("\nVerification:")
    print(f"Expected Value for [0,0]: {expected_val}")
    print(f"Actual Value for [0,0]:   {actual_val}")

    if expected_val == actual_val:
        print(">> Value Match: SUCCESS")
    else:
        print(">> Value Match: FAILED")

    # 检查整体是否符合
    # f_selected 的第 k 个维度的值应该等于 indices[:, k]
    is_correct = True
    for b in range(B):
        for k in range(K):
            target_idx = indices[b, k]
            # 取该切片的平均值，应该等于 target_idx (因为我们造数据时就是这么填的)
            slice_mean = f_selected[b, k].mean().item()
            if not abs(slice_mean - target_idx) < 1e-5:
                is_correct = False
                print(f"Mismatch at Batch {b}, K {k}")

    if is_correct:
        print(">> All batch/channel consistencies verified.")
