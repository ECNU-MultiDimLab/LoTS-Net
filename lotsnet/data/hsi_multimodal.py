import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from tqdm import tqdm
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

# 文本输入模式（控制两个支路的文本分配方式）
TEXT_MODE_CHOICES = ("default", "merge", "spatial_only", "spectral_only")

# 各文本编码器输出的嵌入维度
TEXT_ENCODER_DIM_MAP: Dict[str, int] = {
    "default": 1024,  # 当前默认文件（biobert-large-cased）
    "biobert-large-cased": 1024,
    "biobert-base-cased": 768,  # base 模型为 768 维
    "bert-large-cased": 1024,
}

# 各文本编码器对应的文件名后缀（"" 表示默认，即无后缀）
TEXT_ENCODER_SUFFIX_MAP: Dict[str, str] = {
    "default": "",
    "biobert-large-cased": "_biobert-large-cased",
    "biobert-base-cased": "_biobert-base-cased",
    "bert-large-cased": "_bert-large-cased",
}


class HSIMultimodalDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        transform=None,
        calculate_stats=True,
        max_seq_len_txt=50,
        text_encoder: str = "default",  # 选择使用哪种文本编码器的预计算嵌入
        text_mode: str = "default",  # 控制两个支路的文本分配方式
    ):
        if text_encoder not in TEXT_ENCODER_DIM_MAP:
            raise ValueError(
                f"text_encoder 必须为 {list(TEXT_ENCODER_DIM_MAP.keys())} 之一，"
                f"但收到了 {text_encoder!r}。"
            )
        if text_mode not in TEXT_MODE_CHOICES:
            raise ValueError(
                f"text_mode 必须为 {list(TEXT_MODE_CHOICES)} 之一，"
                f"但收到了 {text_mode!r}。"
            )
        self.data_root = data_root
        self.transform = transform
        self.max_seq_len_txt = max_seq_len_txt
        self.text_encoder = text_encoder
        self.text_dim = TEXT_ENCODER_DIM_MAP[text_encoder]  # 供外部（模型初始化）读取
        self.text_mode = text_mode

        # 1. 路径定义
        self.img_dir = os.path.join(data_root, "images")
        self.msk_dir = os.path.join(data_root, "masks")
        self.txt_dir = os.path.join(data_root, "texts")

        for d in [self.img_dir, self.msk_dir, self.txt_dir]:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Directory structure error! Missing: {d}")

        # 2. 扫描目录并建立映射
        img_paths = glob.glob(os.path.join(self.img_dir, "*.npy"))
        msk_paths = glob.glob(os.path.join(self.msk_dir, "*.npy"))
        txt_paths = glob.glob(os.path.join(self.txt_dir, "*.pt"))

        img_map = {os.path.splitext(os.path.basename(p))[0]: p for p in img_paths}
        msk_map = {os.path.splitext(os.path.basename(p))[0]: p for p in msk_paths}
        # 只用默认（无后缀）的 .pt 文件来确定 common_names，
        # 带编码器后缀的文件通过字符串拼接构造路径，不参与集合求交。
        txt_map = {
            os.path.splitext(os.path.basename(p))[0].replace("_embeddings", ""): p
            for p in txt_paths
            if not any(
                os.path.splitext(os.path.basename(p))[0].endswith(sfx)
                for sfx in TEXT_ENCODER_SUFFIX_MAP.values()
                if sfx  # 跳过空字符串（default）
            )
        }

        # 3. 匹配文件
        common_names = sorted(list(img_map.keys() & msk_map.keys() & txt_map.keys()))

        # 4. 生成数据列表：第三个元素改为路径字典，支持多种编码器
        self.data_list: List[Tuple[str, str, Dict[str, str], str]] = []
        for name in common_names:
            txt_path_dict = {
                enc: os.path.join(self.txt_dir, f"{name}{sfx}.pt")
                for enc, sfx in TEXT_ENCODER_SUFFIX_MAP.items()
            }
            self.data_list.append((img_map[name], msk_map[name], txt_path_dict, name))

        # 5. 打印统计信息 (略，保持原有逻辑)
        if len(self.data_list) == 0:
            raise RuntimeError("No matched {Image, Mask, Text} triplets found!")

        # 6. 计算统计数据 (略，保持原有逻辑)
        if calculate_stats:
            self._scan_dataset_statistics()

    def _scan_dataset_statistics(self):
        """
        遍历数据集，计算图像均值方差、全局像素极值和 Mask 类别分布。
        """
        print("\n[Dataset] Scanning data for statistics...")

        channel_sum = None
        channel_sq_sum = None
        total_pixels = 0
        num_channels = 0
        global_min = np.inf
        global_max = -np.inf
        class_counts = defaultdict(int)
        total_mask_pixels = 0

        # 只遍历前三个元素 (img, msk, txt, name) -> (img, msk)
        for img_path, msk_path, _, _ in tqdm(self.data_list, desc="Calculating Stats"):
            try:
                img_mmap = np.load(img_path, mmap_mode="r")
                if img_mmap.ndim == 2:
                    img_mmap = img_mmap[..., None]
                h, w, c = img_mmap.shape

                if channel_sum is None:
                    num_channels = c
                    channel_sum = np.zeros(c, dtype=np.float64)
                    channel_sq_sum = np.zeros(c, dtype=np.float64)

                if c != num_channels:
                    continue

                flat_img = img_mmap.reshape(-1, c)
                channel_sum += np.sum(flat_img, axis=0)
                channel_sq_sum += np.sum(flat_img**2, axis=0)
                total_pixels += h * w
                global_min = min(global_min, float(np.min(img_mmap)))
                global_max = max(global_max, float(np.max(img_mmap)))

                # Mask
                mask_mmap = np.load(msk_path, mmap_mode="r")
                unique, counts = np.unique(mask_mmap, return_counts=True)
                for u, cnt in zip(unique, counts):
                    class_counts[u] += cnt
                total_mask_pixels += mask_mmap.size

            except Exception as e:
                print(f"Error scanning {img_path}: {e}")
                continue

        if total_pixels > 0:
            self.img_mean = channel_sum / total_pixels
            self.img_var = (channel_sq_sum / total_pixels) - (self.img_mean**2)
            self.img_global_min = global_min
            self.img_global_max = global_max

        self.class_distribution = {}
        sorted_classes = sorted(class_counts.keys())

        print("\n" + "=" * 50)
        print("Dataset Statistics Report")
        if num_channels > 0:
            print(f"Mean (Global Avg): {np.mean(self.img_mean):.4f}")
            print(f"Std  (Global Avg): {np.mean(np.sqrt(self.img_var)):.4f}")
            if total_pixels > 0:
                print(f"Min  (Global):     {self.img_global_min:.6f}")
                print(f"Max  (Global):     {self.img_global_max:.6f}")
        print("-" * 50)
        print(f"{'Class ID':<10} | {'Count':<15} | {'Ratio':<10}")
        for cls_id in sorted_classes:
            ratio = class_counts[cls_id] / total_mask_pixels
            self.class_distribution[cls_id] = ratio
            print(f"{cls_id:<10} | {class_counts[cls_id]:<15} | {ratio:.2%}")
        print("=" * 50 + "\n")

    def compute_fg_bg_ratio(self) -> list:
        """
        计算数据集的前景（class 1）和背景（class 0）像素比例，
        返回 [fg_ratio, bg_ratio]，可直接作为 compute_metrics 的 class_weights 参数。

        约定与 compute_metrics 对齐：
            class_weights[0] = 前景权重（fg_ratio）
            class_weights[1] = 背景权重（bg_ratio）

        优先复用 _scan_dataset_statistics 已计算的 class_distribution（无额外 I/O）；
        若尚未统计（calculate_stats=False），则按需扫描全量 mask。

        Returns:
            [fg_ratio, bg_ratio]，两者之和为 1.0。
        """
        # 复用已有统计结果（_scan_dataset_statistics 的输出）
        if hasattr(self, "class_distribution") and self.class_distribution:
            fg_ratio = float(self.class_distribution.get(1, 0.0))
            bg_ratio = float(self.class_distribution.get(0, 1.0))
            total = fg_ratio + bg_ratio
            if total > 0:
                return [fg_ratio / total, bg_ratio / total]

        # 按需扫描（calculate_stats=False 时走此路径）
        total_fg = 0
        total_bg = 0
        for _, msk_path, _, _ in tqdm(self.data_list, desc="Computing FG/BG ratio"):
            mask = np.load(msk_path, mmap_mode="r")
            total_fg += int((mask == 1).sum())
            total_bg += int((mask == 0).sum())

        total = total_fg + total_bg
        if total == 0:
            return [0.5, 0.5]
        fg_ratio = total_fg / total
        bg_ratio = total_bg / total

        print(
            f"\n[FG/BG Ratio] fg={fg_ratio*100:.1f}%  bg={bg_ratio*100:.1f}%"
            f"  (bg:fg ≈ {bg_ratio/fg_ratio:.2f}:1)"
        )
        return [fg_ratio, bg_ratio]

    def get_case_ids(self) -> list:
        """返回所有样本的 case_id 列表，顺序与 data_list / __getitem__ index 一致。"""
        return [item[3] for item in self.data_list]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, msk_path, txt_path_dict, case_id = self.data_list[idx]
        txt_path = txt_path_dict[self.text_encoder]

        # 1. 加载 HSI
        try:
            img = np.load(img_path)  # [H, W, C]
            if img.ndim == 2:
                img = img[..., None]
            img = img.transpose(2, 0, 1)  # HWC -> CHW
            img = torch.from_numpy(img).float()
        except Exception as e:
            raise IOError(f"Failed to load image: {img_path}") from e

        # 2. 加载 Mask
        try:
            msk = np.load(msk_path)
            msk = torch.from_numpy(msk.astype(np.int64)).long()
        except Exception as e:
            raise IOError(f"Failed to load mask: {msk_path}") from e

        # 3. 加载 Text Embeddings
        try:
            text_data = torch.load(txt_path, map_location="cpu", weights_only=True)
            text_spec = text_data["spectral_evolution"]  # [L1, text_dim]
            text_spa = text_data["spatial_texture"]  # [L2, text_dim]

            # 文本模式：控制两个支路分别收到哪种文本
            #   default      : 光谱支路用光谱文本，空间支路用空间文本（原始行为）
            #   merge        : 两个支路都使用光谱+空间拼接文本
            #   spatial_only : 两个支路都使用空间文本
            #   spectral_only: 两个支路都使用光谱文本
            if self.text_mode == "merge":
                merged = torch.cat([text_spec, text_spa], dim=0)  # [L1+L2, text_dim]
                text_spec = merged
                text_spa = merged
            elif self.text_mode == "spatial_only":
                text_spec = text_spa  # 光谱支路也使用空间文本
            elif self.text_mode == "spectral_only":
                text_spa = text_spec  # 空间支路也使用光谱文本
            # text_mode == "default"：不做任何改动

            # 拼接并处理 text_spa_spec（用于定长文本输入的模型，基于调整后的文本）
            text_concat = torch.cat([text_spec, text_spa], dim=0)
            current_len = text_concat.shape[0]
            if current_len > self.max_seq_len_txt:
                text_spa_spec = text_concat[: self.max_seq_len_txt]
            elif current_len < self.max_seq_len_txt:
                padding = torch.zeros(self.max_seq_len_txt - current_len, self.text_dim)
                text_spa_spec = torch.cat([text_concat, padding], dim=0)
            else:
                text_spa_spec = text_concat

        except Exception:
            print(f"[Warning] Failed to load text: {txt_path}, using zero placeholder.")
            text_spec = torch.zeros(1, self.text_dim)
            text_spa = torch.zeros(1, self.text_dim)
            text_spa_spec = torch.zeros(self.max_seq_len_txt, self.text_dim)

        return {
            "image": img,
            "mask": msk,
            "text_spec": text_spec,
            "text_spa": text_spa,
            "text_spa_spec": text_spa_spec,  # [新增]
            "id": case_id,
        }


def multimodal_collate_fn(batch):
    """
    更新后的 Collate Function
    """
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    ids = [item["id"] for item in batch]

    # 变长文本处理 (TeSS-Net 使用)
    spec_list = [item["text_spec"] for item in batch]
    spa_list = [item["text_spa"] for item in batch]
    text_spec_padded = pad_sequence(spec_list, batch_first=True, padding_value=0.0)
    text_spa_padded = pad_sequence(spa_list, batch_first=True, padding_value=0.0)

    # 定长文本处理 (LViT 使用)
    # text_spa_spec 已经是固定长度了，可以直接 stack
    text_spa_spec_stacked = torch.stack([item["text_spa_spec"] for item in batch])

    return {
        "image": images,
        "mask": masks,
        "text_spec": text_spec_padded,
        "text_spa": text_spa_padded,
        "text_spa_spec": text_spa_spec_stacked,  # [新增]
        "id": ids,
    }


def compute_fg_bg_ratio_for_subset(subset) -> list:
    """
    对 torch.utils.data.Subset（或直接是 HSIMultimodalDataset）计算实际的
    前景/背景像素比例，返回 [fg_ratio, bg_ratio]。

    与 HSIMultimodalDataset.compute_fg_bg_ratio() 不同：
      - 该方法只扫描 subset 内部的样本，而非全量数据集
      - 适合在数据集划分完成后，针对 val 子集或 test 子集分别计算实际比例
      - 通常 val/test 子集的比例与全集平均存在偏差（特别是 CSV 偏置划分时），
        用本函数得到的比例作为 metric_class_weights 更准确

    Args:
        subset: torch.utils.data.Subset 或 HSIMultimodalDataset 实例。
                Subset 的底层 dataset 必须是 HSIMultimodalDataset。

    Returns:
        [fg_ratio, bg_ratio]，两者之和为 1.0。
    """
    # 确定底层 dataset 和要遍历的 index 列表
    if hasattr(subset, "indices"):
        # torch.utils.data.Subset
        base_dataset = subset.dataset
        indices = subset.indices
    elif hasattr(subset, "data_list"):
        # 直接是 HSIMultimodalDataset
        base_dataset = subset
        indices = range(len(subset.data_list))
    else:
        raise TypeError(
            f"compute_fg_bg_ratio_for_subset: 不支持的类型 {type(subset).__name__}，"
            "需要 HSIMultimodalDataset 或其 Subset。"
        )

    total_fg = 0
    total_bg = 0
    for idx in tqdm(indices, desc="Computing subset FG/BG ratio", leave=False):
        _, msk_path, _, _ = base_dataset.data_list[idx]
        mask = np.load(msk_path, mmap_mode="r")
        total_fg += int((mask == 1).sum())
        total_bg += int((mask == 0).sum())

    total = total_fg + total_bg
    if total == 0:
        return [0.5, 0.5]

    fg_ratio = total_fg / total
    bg_ratio = total_bg / total
    return [fg_ratio, bg_ratio]


def load_split_from_csv(dataset: "HSIMultimodalDataset", csv_path: str) -> dict:
    """
    从 CSV 文件读取预设的数据集划分，返回 split_name → index_list 的字典。

    CSV 格式要求（由 dataview.ipynb 生成）：
        case_id  : 样本唯一标识，与 HSIMultimodalDataset.data_list[i][3] 对应
        split    : 划分名称，全监督为 train/val/test，
                   半监督为 train_labeled/train_unlabeled/val/test

    Args:
        dataset  : HSIMultimodalDataset 实例（已完成 __init__，data_list 已建立）
        csv_path : CSV 文件路径

    Returns:
        dict，例如 {"train": [0, 3, 5, ...], "val": [1, 7, ...], "test": [2, ...]}
        索引对应 dataset[i] 的 i，可直接传给 torch.utils.data.Subset。

    Raises:
        FileNotFoundError : csv_path 不存在
        KeyError          : CSV 缺少 'case_id' 或 'split' 列
        ValueError        : CSV 中的 case_id 与当前数据集完全无交集
    """
    import csv as _csv

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"split CSV 不存在: {csv_path}")

    # 建立 case_id → dataset index 映射
    id_to_idx = {item[3]: i for i, item in enumerate(dataset.data_list)}

    splits: dict = {}
    missing: list = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        if "case_id" not in (reader.fieldnames or []) or "split" not in (
            reader.fieldnames or []
        ):
            raise KeyError(
                f"CSV 缺少必要列，需包含 'case_id' 和 'split'，"
                f"实际列名: {reader.fieldnames}"
            )
        for row in reader:
            cid = row["case_id"].strip()
            sp = row["split"].strip()
            if cid not in id_to_idx:
                missing.append(cid)
                continue
            splits.setdefault(sp, []).append(id_to_idx[cid])

    if missing:
        print(
            f"[load_split_from_csv] Warning: {len(missing)} case_id(s) not found in dataset "
            f"(first 5: {missing[:5]})"
        )

    if not splits:
        raise ValueError(
            f"CSV 中的所有 case_id 均未在数据集中找到，请检查 csv_path 与 data_root 是否匹配。"
        )

    total = sum(len(v) for v in splits.values())
    print(f"[load_split_from_csv] Loaded from '{csv_path}':")
    for sp in sorted(splits):
        print(
            f"  {sp:20s}: {len(splits[sp]):4d} samples  ({len(splits[sp])/total*100:.1f}%)"
        )

    return splits


# === 测试代码 ===
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # 假设你的数据根目录结构如下
    # root/
    #   images/
    #   masks/
    #   texts/

    # 替换为你的真实路径进行测试
    TEST_ROOT = "data/MDC_BIG_L_Resized_512_640_to_256_256_overlap_128_160_filtered"

    # 只有当路径存在时才运行测试
    if os.path.exists(TEST_ROOT):
        print("Testing HSIMultimodalDataset...")

        # 1. 实例化 Dataset
        ds = HSIMultimodalDataset(TEST_ROOT, calculate_stats=False)
        print(f"Dataset size: {len(ds)}")

        # 2. 测试 __getitem__
        item0 = ds[0]
        print("\nItem 0 shapes:")
        print(f"  Image: {item0['image'].shape}")
        print(f"  Mask:  {item0['mask'].shape}")
        print(f"  Text Spec: {item0['text_spec'].shape}")
        print(f"  Text Spa:  {item0['text_spa'].shape}")

        # 3. 测试 DataLoader (Collate Function)
        loader = DataLoader(
            ds, batch_size=4, shuffle=True, collate_fn=multimodal_collate_fn
        )

        batch = next(iter(loader))
        print("\nBatch shapes (Batch Size = 1):")
        print(f"  Image: {batch['image'].shape}")  # Expect [4, C, H, W]
        print(f"  Mask:  {batch['mask'].shape}")  # Expect [4, H, W]
        print(f"  Text Spec: {batch['text_spec'].shape}")  # Expect [4, L_max1, 1024]
        print(f"  Text Spa:  {batch['text_spa'].shape}")  # Expect [4, L_max2, 1024]

    else:
        print("Test Skipped: Path not found.")
