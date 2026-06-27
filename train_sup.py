import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, DistributedSampler

from lotsnet.models.lotsnet import LotsNet
from torch.utils.data import Subset
from lotsnet.data.hsi_multimodal import (
    HSIMultimodalDataset,
    multimodal_collate_fn,
    load_split_from_csv,
    compute_fg_bg_ratio_for_subset,
)
from lotsnet.trainers.sup_ddp import train_supervised_multimodal_ddp
from lotsnet.utils.dist import (
    init_distributed_mode,
    cleanup,
    is_main_process,
    master_print,
)


def _format_arg_value(value):
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (list, tuple)):
        return str(list(value))
    return str(value)


def print_parser_args(parser, args, *, runtime=None):
    sep = "=" * 60
    print(sep)
    print("EXPERIMENT HYPERPARAMETERS")
    print(sep)
    for action in parser._actions:
        if action.dest in (None, "help"):
            continue
        name = action.option_strings[-1].lstrip("-") if action.option_strings else action.dest
        value = getattr(args, action.dest, action.default)
        print(f"  {name:<24}: {_format_arg_value(value)}")
    if runtime:
        print("  " + "-" * 40)
        print("  [runtime / DDP]")
        for key, value in runtime.items():
            print(f"  {key:<24}: {_format_arg_value(value)}")
    print(sep)


def build_parser():
    parser = argparse.ArgumentParser(description="LoTS-Net Fully-Supervised DDP Training")
    parser.add_argument("--dist_url", default="env://")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./records/sup")
    parser.add_argument("--exp_name", type=str, default="LoTSNet_Sup_DDP")

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument(
        "--split_csv",
        type=str,
        default="",
        help=(
            "Path to a pre-generated CSV split file (from dataview.ipynb). "
            "CSV must have columns 'case_id' and 'split' (values: train/val/test). "
            "When set, overrides --val_ratio and --test_ratio. "
            "Leave empty (default) to use random splitting."
        ),
    )
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--img_size", type=int, nargs="+", default=[256, 256])

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help=(
            "Early stopping patience: stop training if Val Dice(all) does not improve "
            "for this many consecutive epochs. "
            "0 (default) disables early stopping entirely."
        ),
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.2,
        help=(
            "Fraction of total epochs used for linear LR warmup (default: 0.2). "
            "After warmup, cosine annealing takes the remaining epochs."
        ),
    )
    parser.add_argument(
        "--best_window_start",
        type=int,
        default=50,
        help="Start epoch (inclusive) of the validation window for 'window best' model saving.",
    )
    parser.add_argument(
        "--best_window_end",
        type=int,
        default=100,
        help="End epoch (inclusive) of the validation window for 'window best' model saving.",
    )
    parser.add_argument(
        "--dice_class_weights",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Dice loss per-class weights (length = num_classes). "
            "None = equal weights for all classes (default). "
            "Examples: --dice_class_weights 0.0 1.0  (foreground-only Dice). "
            "          --dice_class_weights 0.5 1.0  (foreground 2x background). "
            "Weights are L1-normalized internally."
        ),
    )
    parser.add_argument(
        "--ce_class_weights",
        type=float,
        nargs="+",
        default=None,
        help=(
            "CrossEntropy loss per-class weights (length = num_classes). "
            "None = equal weight [1.0, 1.0] (default). "
            "Examples: --ce_class_weights 1.0 6.0  (inverse-frequency for ~15%% fg). "
            "Tip: set to (1-fg_ratio)/fg_ratio for inverse-frequency weighting."
        ),
    )
    parser.add_argument(
        "--ce_weight",
        type=float,
        default=1.0,
        help="Overall multiplier for CE loss term in JointSegLoss (default: 1.0).",
    )
    parser.add_argument(
        "--dice_weight",
        type=float,
        default=1.0,
        help="Overall multiplier for Dice loss term in JointSegLoss (default: 1.0).",
    )
    parser.add_argument(
        "--lambda_aux_sup",
        type=float,
        default=0.6,
        help=(
            "Weight for auxiliary spectral head loss in supervised training "
            "(loss = loss_main + lambda_aux_sup * loss_aux). Default: 0.6. "
            "Set to 0.0 to disable the auxiliary head loss entirely."
        ),
    )

    parser.add_argument("--in_chans", type=int, default=60)
    parser.add_argument("--text_dim", type=int, default=1024)
    parser.add_argument("--c_spe", type=int, default=64)
    parser.add_argument("--c_attn", type=int, default=64)
    parser.add_argument("--router_top_k", type=int, default=10)
    parser.add_argument("--stem_ch", type=int, default=64)
    parser.add_argument("--queue_len", type=int, default=100)
    parser.add_argument("--queue_device", type=str, default="cpu")
    parser.add_argument("--smd_rank", type=int, default=16)
    parser.add_argument("--smd_steps", type=int, default=6)
    parser.add_argument(
        "--router_temperature",
        type=float,
        default=1.0,
        help=(
            "Softmax temperature applied to Top-K router scores before aggregation. "
            "<1.0 sharpens weights (more decisive), >1.0 softens them (more uniform). "
            "Default: 1.0. Recommended starting range: 0.5 ~ 2.0."
        ),
    )
    parser.add_argument(
        "--smd_ema_bases",
        action="store_true",
        help=(
            "Use persistent EMA-updated NMF bases instead of random initialization "
            "each forward pass (rand_init=False + update_during_train=True). "
            "STRONGLY RECOMMENDED: fixes incorrect gradients caused by gradient "
            "checkpointing + rand_init, and stabilizes BatchNorm/GroupNorm statistics."
        ),
    )

    parser.add_argument(
        "--progress",
        type=str,
        default="tqdm",
        choices=["tqdm", "log", "none"],
        help="Step-level output: tqdm progress bar, periodic log lines, or silent.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="Print one log line every N steps when --progress=log.",
    )


    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default="top1",
        choices=["top1", "chunked"],
        help="Queue retrieval: 'top1' (paper Eq.1, low VRAM) or 'chunked' (dense attention, chunked).",
    )
    parser.add_argument(
        "--queue_chunk_size",
        type=int,
        default=32,
        help="Entries per chunk when --retrieval_mode=chunked.",
    )


    parser.add_argument(
        "--find_unused_parameters",
        action="store_true",
        help="Enable DDP find_unused_parameters (may add overhead; disable if no unused params).",
    )

    return parser


def get_args():
    parser = build_parser()
    return parser.parse_args(), parser


def main():
    args, parser = get_args()
    init_distributed_mode(args)
    device = torch.device(args.gpu)


    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    seed = args.seed + args.rank
    torch.manual_seed(args.seed)
    np.random.seed(seed)

    if len(args.img_size) == 1:
        args.img_size = (args.img_size[0], args.img_size[0])
    else:
        args.img_size = tuple(args.img_size)

    master_print(f"\n=== LoTS-Net Fully-Supervised DDP (Rank {args.rank}) ===")

    if is_main_process():
        print_parser_args(
            parser,
            args,
            runtime={
                "rank": args.rank,
                "world_size": args.world_size,
                "gpu": args.gpu,
                "distributed": getattr(args, "distributed", False),
            },
        )

    if is_main_process():
        print(">>> Loading Dataset...")

    full_dataset = HSIMultimodalDataset(
        data_root=args.data_root,
        transform=None,
        calculate_stats=is_main_process(),
    )

    if args.split_csv:

        splits = load_split_from_csv(full_dataset, args.split_csv)
        for required in ("train", "val", "test"):
            if required not in splits:
                raise ValueError(
                    f"--split_csv 文件中缺少 '{required}' 划分，"
                    f"实际包含: {list(splits.keys())}"
                )
        train_dataset = Subset(full_dataset, splits["train"])
        val_dataset   = Subset(full_dataset, splits["val"])
        test_dataset  = Subset(full_dataset, splits["test"])
        master_print(
            f"[Split] CSV mode: train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )
    else:

        train_size = int(len(full_dataset) * (1 - args.val_ratio - args.test_ratio))
        val_size   = int(len(full_dataset) * args.val_ratio)
        test_size  = len(full_dataset) - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        master_print(
            f"[Split] Random mode (seed={args.seed}): train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )


    val_fg_bg  = compute_fg_bg_ratio_for_subset(val_dataset)
    test_fg_bg = compute_fg_bg_ratio_for_subset(test_dataset)
    args.val_metric_class_weights  = val_fg_bg
    args.test_metric_class_weights = test_fg_bg

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=False, num_workers=args.num_workers, collate_fn=multimodal_collate_fn,
        drop_last=True, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        shuffle=False, num_workers=args.num_workers, collate_fn=multimodal_collate_fn,
        drop_last=False, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, sampler=test_sampler,
        shuffle=False, num_workers=args.num_workers, collate_fn=multimodal_collate_fn,
        drop_last=False, pin_memory=True,
    )

    model = LotsNet(
        img_size=args.img_size,
        in_chans=args.in_chans,
        num_classes=args.num_classes,
        text_dim=args.text_dim,
        c_spe=args.c_spe,
        c_attn=args.c_attn,
        router_top_k=args.router_top_k,
        stem_ch=args.stem_ch,
        layer_channels=[args.stem_ch * 2, args.stem_ch * 4, args.stem_ch * 4],
        queue_len=args.queue_len,
        queue_device=args.queue_device,
        queue_retrieval_mode=args.retrieval_mode,
        queue_chunk_size=args.queue_chunk_size,
        smd_rank=args.smd_rank,
        smd_steps=args.smd_steps,
        smd_ema_bases=args.smd_ema_bases,
        router_temperature=args.router_temperature,
    ).to(device)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu],
        find_unused_parameters=args.find_unused_parameters,
        broadcast_buffers=False,
    )

    master_print(f"Model initialized. Using {args.world_size} GPUs.")

    train_supervised_multimodal_ddp(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_sampler=train_sampler,
        args=args,
    )

    cleanup()


if __name__ == "__main__":
    main()
