#!/usr/bin/env python
"""Quick dataset smoke test (no DDP / torchrun required)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import torch
from torch.utils.data import DataLoader, random_split

from lotsnet.data.hsi_multimodal import HSIMultimodalDataset, multimodal_collate_fn
from lotsnet.data.hsi_semi import HSISemiDataset, semi_collate_fn


def get_args():
    p = argparse.ArgumentParser(description="Dataset smoke test")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--label_ratio", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = get_args()

    print("=" * 50)
    print("Dataset smoke test")
    print("=" * 50)
    print(f"data_root    : {args.data_root}")
    print(f"label_ratio  : {args.label_ratio}")
    print(f"batch_size   : {args.batch_size}")
    print(f"num_workers  : {args.num_workers}")
    print("=" * 50)

    full_dataset = HSIMultimodalDataset(
        data_root=args.data_root,
        calculate_stats=False,
    )
    print(f"\n[OK] HSIMultimodalDataset loaded: {len(full_dataset)} samples")

    n = len(full_dataset)
    labeled_size = int(n * args.label_ratio)
    unlabeled_size = n - labeled_size
    labeled_set, unlabeled_set = random_split(
        full_dataset,
        [labeled_size, unlabeled_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"[OK] Split: labeled={len(labeled_set)}, unlabeled={len(unlabeled_set)}")

    semi_dataset = HSISemiDataset(unlabeled_set)
    print(f"[OK] HSISemiDataset wrapped: {len(semi_dataset)} samples")

    l_loader = DataLoader(
        labeled_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=multimodal_collate_fn,
    )
    u_loader = DataLoader(
        semi_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=semi_collate_fn,
    )

    batch_l = next(iter(l_loader))
    batch_u = next(iter(u_loader))

    print(f"\n[Labeled batch]")
    print(f"  image     : {batch_l['image'].shape}")
    print(f"  mask      : {batch_l['mask'].shape}")
    print(f"  text_spec : {batch_l['text_spec'].shape}")
    print(f"  text_spa  : {batch_l['text_spa'].shape}")

    print(f"\n[Unlabeled batch]")
    print(f"  image_w   : {batch_u['image_w'].shape}")
    print(f"  image_s   : {batch_u['image_s'].shape}")
    print(f"  text_spec : {batch_u['text_spec'].shape}")
    print(f"  text_spa  : {batch_u['text_spa'].shape}")

    img_ch = batch_l["image"].shape[1]
    assert img_ch in {40, 60}, f"unexpected channel count: {img_ch}"
    mask_vals = batch_l["mask"].unique().tolist()
    print(f"\n[Stats]")
    print(f"  spectral bands : {img_ch}")
    print(f"  mask values    : {mask_vals}")

    print("\n" + "=" * 50)
    print("All checks passed.")
    print("=" * 50)


if __name__ == "__main__":
    main()
