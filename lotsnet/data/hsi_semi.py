import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# =================================================================
# 1. 高光谱数据强弱增强器 (HSI Weak-Strong Augmentation)
# =================================================================
class HSIWeakStrongAugmentation:
    def __init__(self, noise_std=0.05, jitter_range=0.2):
        self.noise_std = noise_std
        self.jitter_range = jitter_range

    def weak_aug(self, x):
        """
        弱增强：仅作基础的张量转换和归一化 (假定输入已归一化)
        x: [C, H, W] Tensor
        """
        return x.clone()

    def strong_aug(self, x):
        """
        强增强：颜色抖动 + 高斯噪声
        x: [C, H, W] Tensor
        """
        x_strong = x.clone()

        # 1. 颜色抖动 (Color Jitter: alpha * x + beta)
        # 从 U[1-jitter, 1+jitter] 采样 alpha
        alpha = np.random.uniform(1.0 - self.jitter_range, 1.0 + self.jitter_range)
        # 从 U[-0.1, 0.1] 采样 beta
        beta = np.random.uniform(-0.1, 0.1)

        x_strong = alpha * x_strong + beta

        # 2. 加性高斯噪声
        noise = torch.randn_like(x_strong) * self.noise_std
        x_strong = x_strong + noise

        # 限制在合理范围内 (防止过曝，假设原数据在 0-1 之间，如果有负数可忽略 clamp)
        # 如果你的数据没有严格归一化到 [0,1]，请注释掉下面这行
        # x_strong = torch.clamp(x_strong, 0.0, 1.0)

        return x_strong


# =================================================================
# 2. 半监督无标签数据集类 (HSI Semi Dataset)
# =================================================================
class HSISemiDataset(Dataset):
    def __init__(self, subset_dataset, max_seq_len_txt=50):
        """
        包装器 Dataset：接收一个 Subset（来自 random_split），并返回强/弱双视图

        Args:
            subset_dataset: 原始 HSIMultimodalDataset 的 Subset 对象
            max_seq_len_txt: 文本最大长度，用于定长模型
        """
        self.dataset = subset_dataset
        self.augmentor = HSIWeakStrongAugmentation()
        self.max_seq_len_txt = max_seq_len_txt

    def compute_fg_bg_ratio(self) -> list:
        """
        委托给底层 HSIMultimodalDataset（穿透 Subset 包装层）。
        返回 [fg_ratio, bg_ratio]，与 HSIMultimodalDataset.compute_fg_bg_ratio() 一致。
        """
        base = self.dataset
        # Subset 对象通过 .dataset 属性指向原始 Dataset，可能嵌套多层
        while hasattr(base, "dataset"):
            base = base.dataset
        if hasattr(base, "compute_fg_bg_ratio"):
            return base.compute_fg_bg_ratio()
        raise AttributeError(
            f"底层数据集 {type(base).__name__} 不支持 compute_fg_bg_ratio()，"
            "请确认底层为 HSIMultimodalDataset。"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 1. 从底层的 Dataset 取出数据
        # 这里的 item 是一个 dict，包含了 image, mask, text_spec, text_spa 等
        item = self.dataset[idx]

        orig_img = item["image"]  # [C, H, W]

        # 2. 生成双视图 (Dual Views)
        img_w = self.augmentor.weak_aug(orig_img)
        img_s = self.augmentor.strong_aug(orig_img)

        # 3. 构造返回字典
        # 半监督无标签数据理论上不需要 mask，但为了代码兼容或 Debug，我们可以保留
        out_dict = {
            "image_w": img_w,
            "image_s": img_s,
            "text_spec": item["text_spec"],
            "text_spa": item["text_spa"],
            "id": item["id"],
            # "mask": item['mask'] # 如果 Trainer 不需要，可以不返回
        }

        # 处理定长拼接文本 (兼容 LViT / ARSeg 等单文本模型)
        if "text_spa_spec" in item:
            out_dict["text_spa_spec"] = item["text_spa_spec"]

        return out_dict


# =================================================================
# 3. 半监督 Collate Function
# =================================================================
def semi_collate_fn(batch):
    """
    用于无标签数据的 DataLoader，处理变长文本，并分离 img_w 和 img_s
    """
    # 1. 堆叠图像视图
    images_w = torch.stack([item["image_w"] for item in batch])
    images_s = torch.stack([item["image_s"] for item in batch])
    ids = [item["id"] for item in batch]

    # 2. 处理变长文本 (TeSS-Net 使用)
    spec_list = [item["text_spec"] for item in batch]
    spa_list = [item["text_spa"] for item in batch]

    text_spec_padded = pad_sequence(spec_list, batch_first=True, padding_value=0.0)
    text_spa_padded = pad_sequence(spa_list, batch_first=True, padding_value=0.0)

    out_dict = {
        "image_w": images_w,
        "image_s": images_s,
        "text_spec": text_spec_padded,
        "text_spa": text_spa_padded,
        "id": ids,
    }

    # 3. 处理定长文本 (LViT / ARSeg 等单文本模型使用)
    if "text_spa_spec" in batch[0]:
        text_spa_spec_stacked = torch.stack([item["text_spa_spec"] for item in batch])
        out_dict["text_spa_spec"] = text_spa_spec_stacked

    return out_dict


# === 测试代码 ===
# if __name__ == "__main__":
#     from torch.utils.data import DataLoader
#     from datasets.hsi_multimodal_dataset import (
#         HSIMultimodalDataset,
#     )  # 导入你的基础数据集

#     TEST_ROOT = "./data/MDC_Resized_256_320_to_256_256_overlap_0_0"

#     if os.path.exists(TEST_ROOT):
#         print("Testing HSISemiDataset...")
#         # 1. 实例化基础数据集
#         base_ds = HSIMultimodalDataset(TEST_ROOT, calculate_stats=False)

#         # 2. 用 SemiDataset 包装它 (假设全量数据当做无标签数据测试)
#         semi_ds = HSISemiDataset(base_ds)
#         print(f"Semi Dataset size: {len(semi_ds)}")

#         # 3. 测试 DataLoader
#         loader = DataLoader(
#             semi_ds, batch_size=4, shuffle=True, collate_fn=semi_collate_fn
#         )

#         batch = next(iter(loader))
#         print("\nBatch shapes (Batch Size = 4):")
#         print(f"  Image Weak:   {batch['image_w'].shape}")  # Expect [4, C, H, W]
#         print(f"  Image Strong: {batch['image_s'].shape}")  # Expect [4, C, H, W]
#         print(f"  Text Spec:    {batch['text_spec'].shape}")
#         print(f"  Text Spa:     {batch['text_spa'].shape}")

#         # 验证弱增强和强增强确实不同
#         diff = torch.abs(batch["image_w"] - batch["image_s"]).sum()
#         print(
#             f"\nDifference between Weak and Strong views (should be > 0): {diff.item():.4f}"
#         )
#         assert diff > 0, "Augmentation failed! Weak and Strong are identical."
#         print(">> Test Passed! ✅")
#     else:
#         print("Test Skipped: Path not found.")
