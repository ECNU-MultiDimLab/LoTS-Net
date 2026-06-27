import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, DistributedSampler, Subset

from lotsnet.models.lotsnet import LotsNet

from lotsnet.data.hsi_semi import HSISemiDataset, semi_collate_fn
from lotsnet.data.hsi_multimodal import (
    HSIMultimodalDataset,
    multimodal_collate_fn,
    load_split_from_csv,
    compute_fg_bg_ratio_for_subset,
)
from lotsnet.trainers.semi_sup_ddp import train_semi_supervised_ddp
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
    parser = argparse.ArgumentParser(description="LoTS-Net Semi-Supervised Training")
    parser.add_argument("--dist_url", default="env://", help="url for DDP")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./records/semi")
    parser.add_argument("--exp_name", type=str, default="Semi_LoTSNet")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument(
        "--split_csv",
        type=str,
        default="",
        help=(
            "Path to a pre-generated CSV split file (from dataview.ipynb). "
            "CSV must have columns 'case_id' and 'split' "
            "(values: train_labeled/train_unlabeled/val/test). "
            "When set, overrides --label_ratio, --val_ratio, --test_ratio. "
            "Leave empty (default) to use random splitting."
        ),
    )
    parser.add_argument("--label_ratio", type=float, default=0.2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--img_size", type=int, nargs="+", default=[256, 256])

    parser.add_argument("--use_l_pl", action="store_true",
                        help="Enable pseudo-label loss on unlabeled data.")
    parser.add_argument("--use_l_con", action="store_true",
                        help="Enable consistency regularization loss on unlabeled data.")

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help=(
            "Early stopping patience: stop if Val Dice does not improve for this many "
            "consecutive epochs. 0 (default) disables early stopping."
        ),
    )
    parser.add_argument(
        "--best_window_start",
        type=int,
        default=50,
        help="Start epoch (inclusive) of the validation window for window-best model saving.",
    )
    parser.add_argument(
        "--best_window_end",
        type=int,
        default=100,
        help="End epoch (inclusive) of the validation window for window-best model saving.",
    )
    parser.add_argument(
        "--dice_class_weights",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Dice loss per-class weights (length = num_classes). "
            "None = equal weights (default). "
            "Examples: --dice_class_weights 0.5 1.0"
        ),
    )
    parser.add_argument(
        "--ce_class_weights",
        type=float,
        nargs="+",
        default=None,
        help=(
            "CrossEntropy loss per-class weights (length = num_classes). "
            "None = equal weight [1.0, 1.0] (default)."
        ),
    )
    parser.add_argument(
        "--ce_weight",
        type=float,
        default=1.0,
        help="Overall multiplier for CE loss term in JointSegLoss.",
    )
    parser.add_argument(
        "--dice_weight",
        type=float,
        default=1.0,
        help="Overall multiplier for Dice loss term in JointSegLoss.",
    )

    parser.add_argument("--lambda_aux", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lambda_max", type=float, default=0.4)
    parser.add_argument("--rampup_ratio", type=float, default=1.0 / 3.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.4)
    parser.add_argument("--plateau_ratio", type=float, default=0.1)
    parser.add_argument("--pl_conf_thresh", type=float, default=0.70)

    parser.add_argument("--num_top_k_batches", type=int, default=20)
    parser.add_argument(
        "--sort_by", type=str, default="dice", choices=["dice", "iou", "hd95"]
    )

    parser.add_argument("--in_chans", type=int, default=60)
    parser.add_argument("--c_spe", type=int, default=256)
    parser.add_argument("--c_attn", type=int, default=256)
    parser.add_argument("--stem_ch", type=int, default=256)
    parser.add_argument("--router_top_k", type=int, default=10)
    parser.add_argument(
        "--router_temperature",
        type=float,
        default=1.0,
        help=(
            "Softmax temperature applied to Top-K router scores before aggregation. "
            "<1.0 sharpens weights (more decisive), >1.0 softens them (more uniform). "
            "Default: 1.0."
        ),
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        default="default",
        choices=["default", "biobert-large-cased", "biobert-base-cased", "bert-large-cased"],
    )
    parser.add_argument(
        "--text_mode",
        type=str,
        default="default",
        choices=["default", "merge", "spatial_only", "spectral_only"],
    )
    parser.add_argument("--queue_len", type=int, default=300)
    parser.add_argument("--smd_rank", type=int, default=16)
    parser.add_argument("--smd_steps", type=int, default=6)
    parser.add_argument("--queue_device", type=str, default="cpu")
    parser.add_argument("--router_score_noise_std", type=float, default=0.0)
    parser.add_argument(
        "--smd_ema_bases",
        action="store_true",
        help=(
            "Use persistent EMA-updated NMF bases instead of random initialization "
            "each forward pass (rand_init=False + update_during_train=True). "
            "STRONGLY RECOMMENDED."
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


def build_model(args):
    return LotsNet(
        img_size=args.img_size,
        in_chans=args.in_chans,
        num_classes=args.num_classes,
        c_spe=args.c_spe,
        c_attn=args.c_attn,
        stem_ch=args.stem_ch,
        layer_channels=[args.stem_ch * 2, args.stem_ch * 4, args.stem_ch * 4],
        router_top_k=args.router_top_k,
        text_dim=args.text_dim,
        queue_len=args.queue_len,
        queue_device=args.queue_device,
        queue_retrieval_mode=args.retrieval_mode,
        queue_chunk_size=args.queue_chunk_size,
        smd_rank=args.smd_rank,
        smd_steps=args.smd_steps,
        router_score_noise_std=args.router_score_noise_std,
        smd_ema_bases=args.smd_ema_bases,
        router_temperature=args.router_temperature,
    )


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

    master_print(f"\n=== LoTS-Net Semi-Supervised DDP (Rank {args.rank}) ===")

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
        calculate_stats=False,
        text_encoder=args.text_encoder,
        text_mode=args.text_mode,
    )
    args.text_dim = full_dataset.text_dim

    if args.split_csv:
        splits = load_split_from_csv(full_dataset, args.split_csv)
        for required in ("train_labeled", "train_unlabeled", "val", "test"):
            if required not in splits:
                raise ValueError(
                    f"--split_csv 文件中缺少 '{required}' 划分，"
                    f"实际包含: {list(splits.keys())}"
                )
        l_set = Subset(full_dataset, splits["train_labeled"])
        u_set = Subset(full_dataset, splits["train_unlabeled"])
        v_set = Subset(full_dataset, splits["val"])
        t_set = Subset(full_dataset, splits["test"])
        master_print(
            f"[Split] CSV mode: labeled={len(l_set)}, unlabeled={len(u_set)}, "
            f"val={len(v_set)}, test={len(t_set)}"
        )
    else:
        val_size       = int(len(full_dataset) * args.val_ratio)
        test_size      = int(len(full_dataset) * args.test_ratio)
        labeled_size   = int(len(full_dataset) * args.label_ratio)
        unlabeled_size = len(full_dataset) - labeled_size - val_size - test_size
        l_set, u_set, v_set, t_set = random_split(
            full_dataset,
            [labeled_size, unlabeled_size, val_size, test_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        master_print(
            f"[Split] Random mode (seed={args.seed}): labeled={len(l_set)}, "
            f"unlabeled={len(u_set)}, val={len(v_set)}, test={len(t_set)}"
        )


    val_fg_bg  = compute_fg_bg_ratio_for_subset(v_set)
    test_fg_bg = compute_fg_bg_ratio_for_subset(t_set)
    args.val_metric_class_weights  = val_fg_bg
    args.test_metric_class_weights = test_fg_bg

    unlabeled_dataset = HSISemiDataset(u_set)

    l_sampler = DistributedSampler(l_set, shuffle=True)
    u_sampler = DistributedSampler(unlabeled_dataset, shuffle=True)

    l_loader = DataLoader(
        l_set, batch_size=args.batch_size, sampler=l_sampler,
        num_workers=args.num_workers, collate_fn=multimodal_collate_fn,
        pin_memory=True,
    )
    u_loader = DataLoader(
        unlabeled_dataset, batch_size=args.batch_size, sampler=u_sampler,
        num_workers=args.num_workers, collate_fn=semi_collate_fn,
        pin_memory=True,
    )
    v_loader = DataLoader(
        v_set, batch_size=args.batch_size,
        sampler=DistributedSampler(v_set, shuffle=False),
        num_workers=args.num_workers, collate_fn=multimodal_collate_fn,
        pin_memory=True,
    )
    t_loader = DataLoader(
        t_set, batch_size=args.batch_size,
        sampler=DistributedSampler(t_set, shuffle=False),
        num_workers=args.num_workers, collate_fn=multimodal_collate_fn,
        pin_memory=True,
    )

    model = build_model(args).to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu],
        find_unused_parameters=args.find_unused_parameters,
        broadcast_buffers=False,
    )

    master_print(f"Model initialized. Using {args.world_size} GPUs.")

    train_semi_supervised_ddp(
        model, l_loader, u_loader, v_loader, t_loader, l_sampler, u_sampler, args
    )
    cleanup()


if __name__ == "__main__":
    main()
