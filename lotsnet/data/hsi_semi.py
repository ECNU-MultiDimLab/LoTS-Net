import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class HSIWeakStrongAugmentation:
    def __init__(self, noise_std=0.05, jitter_range=0.2):
        self.noise_std = noise_std
        self.jitter_range = jitter_range

    def weak_aug(self, x):
        return x.clone()

    def strong_aug(self, x):
        x_strong = x.clone()


        alpha = np.random.uniform(1.0 - self.jitter_range, 1.0 + self.jitter_range)

        beta = np.random.uniform(-0.1, 0.1)

        x_strong = alpha * x_strong + beta


        noise = torch.randn_like(x_strong) * self.noise_std
        x_strong = x_strong + noise


        return x_strong


class HSISemiDataset(Dataset):
    def __init__(self, subset_dataset, max_seq_len_txt=50):
        self.dataset = subset_dataset
        self.augmentor = HSIWeakStrongAugmentation()
        self.max_seq_len_txt = max_seq_len_txt

    def compute_fg_bg_ratio(self) -> list:
        base = self.dataset

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


        item = self.dataset[idx]

        orig_img = item["image"]


        img_w = self.augmentor.weak_aug(orig_img)
        img_s = self.augmentor.strong_aug(orig_img)


        out_dict = {
            "image_w": img_w,
            "image_s": img_s,
            "text_spec": item["text_spec"],
            "text_spa": item["text_spa"],
            "id": item["id"],

        }


        if "text_spa_spec" in item:
            out_dict["text_spa_spec"] = item["text_spa_spec"]

        return out_dict


def semi_collate_fn(batch):

    images_w = torch.stack([item["image_w"] for item in batch])
    images_s = torch.stack([item["image_s"] for item in batch])
    ids = [item["id"] for item in batch]


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


    if "text_spa_spec" in batch[0]:
        text_spa_spec_stacked = torch.stack([item["text_spa_spec"] for item in batch])
        out_dict["text_spa_spec"] = text_spa_spec_stacked

    return out_dict
