import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.amp import autocast

from lotsnet.utils.losses import (
    JointSegLoss,
    compute_metrics,
    AverageMeter,
    save_visualization_4_1,
)
from lotsnet.utils.dist import is_main_process, master_print, reduce_tensor
from lotsnet.utils.progress import wrap_dataloader, maybe_log_step, maybe_set_postfix


# =================================================================
# 1. 动态权重函数 (Gaussian Ramp-up)
# =================================================================
def get_rampup_weight(current_epoch, T_max, lambda_max):
    """
    Gaussian ramp-up with steep coefficient (-5).
    Starts near zero and reaches lambda_max at epoch >= T_max.
    """
    if current_epoch >= T_max:
        return lambda_max
    else:
        return lambda_max * math.exp(-5.0 * (1.0 - current_epoch / T_max) ** 2)


# =================================================================
# 2. 测试指标美化打印（Top-K 排序均值）
# =================================================================
def print_top_k_metrics(per_batch_metrics, num_top_k_batches=5, sort_by="dice"):
    """
    per_batch_metrics 中每条为 (Dice_all, FgDice, IoU, HD95) 四元组。
    """
    if not per_batch_metrics:
        master_print("No test metrics collected.")
        return

    # 列索引：0=Dice(all), 1=FgDice, 2=IoU, 3=HD95
    sort_cfg = {"dice": (0, True), "fgdice": (1, True), "iou": (2, True), "hd95": (3, False)}
    col_idx, reverse = sort_cfg.get(sort_by.lower(), (0, True))
    sorted_m = sorted(per_batch_metrics, key=lambda x: x[col_idx], reverse=reverse)

    n_total = len(sorted_m)
    max_k   = min(num_top_k_batches, n_total)

    def _mean(lst, col):
        return sum(r[col] for r in lst) / len(lst)

    master_print("\n" + "=" * 60)
    master_print("FINAL TEST RESULTS (Averaged across GPUs):")
    master_print(
        f"  [All {n_total:3d} batches]  "
        f"Dice(all): {_mean(sorted_m, 0):.4f}  "
        f"Dice(fg): {_mean(sorted_m, 1):.4f}  "
        f"IoU: {_mean(sorted_m, 2):.4f}  "
        f"HD95: {_mean(sorted_m, 3):.4f}"
    )
    master_print(f"  --- Top-K breakdown (sorted by {sort_by}) ---")
    for k in range(1, max_k + 1):
        top_m = sorted_m[:k]
        master_print(
            f"  [Top-{k:3d}            ]  "
            f"Dice(all): {_mean(top_m, 0):.4f}  "
            f"Dice(fg): {_mean(top_m, 1):.4f}  "
            f"IoU: {_mean(top_m, 2):.4f}  "
            f"HD95: {_mean(top_m, 3):.4f}"
        )
    master_print("=" * 60)


# =================================================================
# 3. 半监督 Trainer 主函数
# =================================================================
def train_semi_supervised_ddp(
    model,
    labeled_loader,
    unlabeled_loader,
    val_loader,
    test_loader,
    labeled_sampler,
    unlabeled_sampler,
    args,
):
    device = torch.device(args.gpu)

    scale_factor = math.sqrt(args.world_size) if args.world_size > 1 else 1.0
    scaled_lr = args.lr * scale_factor
    optimizer = optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=1e-4)

    warmup_ratio  = getattr(args, "warmup_ratio",  0.4)
    plateau_ratio = getattr(args, "plateau_ratio", 0.1)
    warmup_epochs  = max(1, int(args.epochs * warmup_ratio))
    plateau_epochs = max(1, int(args.epochs * plateau_ratio))
    cosine_epochs  = max(1, args.epochs - warmup_epochs - plateau_epochs)

    master_print(
        f"[LR Schedule] warmup={warmup_epochs}ep | "
        f"plateau={plateau_epochs}ep | cosine={cosine_epochs}ep | "
        f"peak_lr={scaled_lr:.2e}"
    )

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs,
    )
    plateau_scheduler = optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, total_iters=plateau_epochs,
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, plateau_scheduler, cosine_scheduler],
        milestones=[warmup_epochs, warmup_epochs + plateau_epochs],
    )

    rampup_ratio   = getattr(args, "rampup_ratio", 1.0 / 3.0)
    T_max_rampup   = max(1, int(args.epochs * rampup_ratio))
    master_print(
        f"[Ramp-up]     rampup={T_max_rampup}ep ({rampup_ratio:.0%} of {args.epochs}ep) | "
        f"lambda_max={args.lambda_max}"
    )

    criterion_seg = JointSegLoss(
        n_classes=args.num_classes,
        ce_weight=getattr(args, "ce_weight", 1.0),
        dice_weight=getattr(args, "dice_weight", 1.0),
        dice_class_weights=getattr(args, "dice_class_weights", None),
        ce_class_weights=getattr(args, "ce_class_weights", None),
    ).to(device)
    criterion_mse = nn.MSELoss().to(device)
    criterion_ce_pl = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    pl_conf_thresh = getattr(args, "pl_conf_thresh", 0.70)

    best_dice = 0.0
    run_save_dir = args.save_dir
    progress = getattr(args, "progress", "tqdm")
    log_interval = getattr(args, "log_interval", 20)
    if is_main_process():
        os.makedirs(run_save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        labeled_sampler.set_epoch(epoch)
        unlabeled_sampler.set_epoch(epoch)

        model.train()
        loss_meter        = AverageMeter()
        sup_loss_meter    = AverageMeter()
        sup_main_meter    = AverageMeter()
        sup_aux_meter     = AverageMeter()
        unsup_loss_meter  = AverageMeter()

        rampup_weight = get_rampup_weight(epoch, T_max_rampup, args.lambda_max)

        iter_labeled = iter(labeled_loader)
        loader_iter = wrap_dataloader(
            unlabeled_loader,
            progress=progress,
            desc=f"Epoch {epoch}/{args.epochs} [Semi-Train]",
            leave=False,
        )

        for step, batch_u in enumerate(loader_iter):
            try:
                batch_l = next(iter_labeled)
            except StopIteration:
                iter_labeled = iter(labeled_loader)
                batch_l = next(iter_labeled)

            optimizer.zero_grad()
            total_loss_val = 0.0

            # ---- Step A: Supervised forward + backward ----
            img_l = batch_l["image"].to(device, non_blocking=True).float()
            mask_l = batch_l["mask"].to(device, non_blocking=True).long()
            text_spec_l = batch_l["text_spec"].to(device, non_blocking=True).float()
            text_spa_l = batch_l["text_spa"].to(device, non_blocking=True).float()

            with model.no_sync():
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    out_l, aux_l = model(img_l, text_spec_l, text_spa_l, input_data_type=0)
                    loss_sup_main = criterion_seg(out_l, mask_l)
                    loss_sup_aux_raw = (
                        criterion_seg(aux_l, mask_l) if aux_l is not None
                        else torch.zeros(1, device=device)
                    )
                    loss_sup = loss_sup_main + args.lambda_aux * loss_sup_aux_raw
                loss_sup.backward()

            sup_loss_val = loss_sup.item()
            total_loss_val += sup_loss_val
            sup_main_meter.update(loss_sup_main.item())
            sup_aux_meter.update(loss_sup_aux_raw.item())
            del img_l, mask_l, text_spec_l, text_spa_l, out_l, aux_l
            del loss_sup, loss_sup_main, loss_sup_aux_raw

            # ---- Step B: Unsupervised forward + backward ----
            unsup_loss_val = 0.0

            if args.use_l_pl or args.use_l_con:
                text_spec_u = batch_u["text_spec"].to(device, non_blocking=True).float()
                text_spa_u = batch_u["text_spa"].to(device, non_blocking=True).float()

                img_u_w = batch_u["image_w"].to(device, non_blocking=True).float()
                with torch.no_grad():
                    with autocast(device_type="cuda", dtype=torch.bfloat16):
                        out_u_w, aux_u_w, f_spa_w, f_ret_w = model(
                            img_u_w, text_spec_u, text_spa_u, input_data_type=1
                        )
                    probs_w = torch.softmax(out_u_w.float(), dim=1)
                    conf_w, pseudo_labels = probs_w.max(dim=1)
                    pseudo_labels = pseudo_labels.clone()
                    pseudo_labels[conf_w < pl_conf_thresh] = -1

                    aux_probs_w = torch.softmax(aux_u_w.float(), dim=1)
                    aux_conf_w, aux_pseudo_labels = aux_probs_w.max(dim=1)
                    aux_pseudo_labels = aux_pseudo_labels.clone()
                    aux_pseudo_labels[aux_conf_w < pl_conf_thresh] = -1

                del img_u_w, out_u_w, aux_u_w, probs_w, conf_w, aux_probs_w, aux_conf_w

                img_u_s = batch_u["image_s"].to(device, non_blocking=True).float()
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    out_u_s, aux_u_s, f_spa_s, f_ret_s = model(
                        img_u_s, text_spec_u, text_spa_u, input_data_type=2
                    )

                    loss_unsup = torch.tensor(0.0, device=device)
                    if args.use_l_pl:
                        loss_pl = criterion_ce_pl(out_u_s, pseudo_labels)
                        loss_unsup += loss_pl
                        if aux_u_s is not None:
                            loss_aux_pl = criterion_ce_pl(aux_u_s, aux_pseudo_labels)
                            loss_unsup += args.lambda_aux * loss_aux_pl

                    if args.use_l_con:
                        loss_con_spa = criterion_mse(f_spa_s, f_spa_w.detach())
                        loss_con_spe = (
                            criterion_mse(f_ret_s, f_ret_w.detach())
                            if (f_ret_s is not None)
                            else 0.0
                        )
                        loss_unsup += args.gamma * (loss_con_spa + loss_con_spe)

                    weighted_loss_unsup = rampup_weight * loss_unsup

                weighted_loss_unsup.backward()

                unsup_loss_val = weighted_loss_unsup.item()
                total_loss_val += unsup_loss_val

                del (
                    img_u_s, out_u_s, aux_u_s, f_spa_s, f_ret_s,
                    pseudo_labels, aux_pseudo_labels, f_spa_w, f_ret_w,
                    loss_unsup, weighted_loss_unsup,
                )

            # ---- Step C: Parameter update ----
            optimizer.step()

            loss_meter.update(total_loss_val, batch_l["image"].size(0))
            sup_loss_meter.update(sup_loss_val)
            if args.use_l_pl or args.use_l_con:
                unsup_loss_meter.update(unsup_loss_val)

            step_msg = (
                f"Tot={loss_meter.avg:.3f}  "
                f"S={sup_loss_meter.avg:.3f}"
                f"(main={sup_main_meter.avg:.3f},aux={sup_aux_meter.avg:.3f})  "
                f"U={unsup_loss_meter.avg:.3f}  W={rampup_weight:.2f}"
            )
            maybe_log_step(
                progress=progress,
                step=step,
                total=len(unlabeled_loader),
                log_interval=log_interval,
                msg=step_msg,
            )
            maybe_set_postfix(
                loader_iter,
                progress=progress,
                Tot=f"{loss_meter.avg:.3f}",
                Main=f"{sup_main_meter.avg:.3f}",
                Aux=f"{sup_aux_meter.avg:.3f}",
                U=f"{unsup_loss_meter.avg:.3f}",
                W=f"{rampup_weight:.2f}",
            )

        scheduler.step()

        # ---- Validation ----
        model.eval()
        val_batch_metrics = []

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device, non_blocking=True).float()
                masks = batch["mask"].to(device, non_blocking=True).long()
                text_spec = batch["text_spec"].to(device, non_blocking=True).float()
                text_spa = batch["text_spa"].to(device, non_blocking=True).float()

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(imgs, text_spec, text_spa)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs

                metrics = compute_metrics(
                    logits, masks, args.num_classes,
                    class_weights=getattr(args, "metric_class_weights", None),
                )
                metrics_tensor = torch.tensor(
                    [metrics["Dice"], metrics["FgDice"],
                     metrics["IoU"], metrics["HD95"]], device=device
                )
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
                metrics_tensor /= args.world_size

                val_batch_metrics.append((
                    metrics_tensor[0].item(),   # Dice(all)
                    metrics_tensor[1].item(),   # FgDice
                    metrics_tensor[2].item(),   # IoU
                    metrics_tensor[3].item(),   # HD95
                ))

        val_dice_avg    = sum(m[0] for m in val_batch_metrics) / len(val_batch_metrics)
        val_fgdice_avg  = sum(m[1] for m in val_batch_metrics) / len(val_batch_metrics)
        val_iou_avg     = sum(m[2] for m in val_batch_metrics) / len(val_batch_metrics)
        val_hd95_avg    = sum(m[3] for m in val_batch_metrics) / len(val_batch_metrics)

        if is_main_process():
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}: "
                f"Loss: {loss_meter.avg:.4f} "
                f"[main={sup_main_meter.avg:.4f}, aux={sup_aux_meter.avg:.4f}] | "
                f"Val Dice(all): {val_dice_avg:.4f} | "
                f"Val Dice(fg): {val_fgdice_avg:.4f} | "
                f"Val IoU: {val_iou_avg:.4f} | "
                f"Val HD95: {val_hd95_avg:.4f} | "
                f"LR: {current_lr:.1e}"
            )
            num_top_k_batches = getattr(args, "num_top_k_batches", 5)
            sort_by           = getattr(args, "sort_by", "dice")
            print(f"  [Val Top-K @ Epoch {epoch}]")
            print_top_k_metrics(val_batch_metrics, num_top_k_batches=num_top_k_batches, sort_by=sort_by)

            save_dict = {
                "epoch": epoch,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(save_dict, os.path.join(run_save_dir, "model_latest.pth"))

            if val_fgdice_avg > best_dice:
                best_dice = val_fgdice_avg
                torch.save(save_dict, os.path.join(run_save_dir, "model_best.pth"))
                print(f">>> New Best Model Saved! FgDice: {best_dice:.4f}")

    # ---- Final Test ----
    dist.barrier()
    master_print("\n" + "=" * 40)
    master_print(">>> Starting Final Testing & Visualization...")

    best_model_path = os.path.join(run_save_dir, "model_best.pth")
    if not os.path.exists(best_model_path):
        master_print(">>> Warning: No best model found!")
        return run_save_dir

    checkpoint = torch.load(best_model_path, map_location=device)
    model.module.load_state_dict(checkpoint["model"])
    model.eval()

    per_batch_metrics = []

    vis_save_dir = None
    if is_main_process():
        vis_save_dir = os.path.join(run_save_dir, "test_visual_results")
        os.makedirs(vis_save_dir, exist_ok=True)

    test_iter = wrap_dataloader(
        test_loader,
        progress=progress,
        desc="Testing",
        leave=True,
    )

    total_saved = 0
    with torch.no_grad():
        for i, batch in enumerate(test_iter):
            imgs = batch["image"].to(device, non_blocking=True).float()
            masks = batch["mask"].to(device, non_blocking=True).long()
            text_spec = batch["text_spec"].to(device, non_blocking=True).float()
            text_spa = batch["text_spa"].to(device, non_blocking=True).float()

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(imgs, text_spec, text_spa)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs

            metrics = compute_metrics(
                logits, masks, args.num_classes,
                class_weights=getattr(args, "metric_class_weights", None),
            )
            batch_metrics = torch.tensor(
                [metrics["Dice"], metrics["FgDice"],
                 metrics["IoU"], metrics["HD95"]], device=device
            )
            dist.all_reduce(batch_metrics, op=dist.ReduceOp.SUM)
            batch_metrics /= args.world_size

            per_batch_metrics.append((
                batch_metrics[0].item(),   # Dice(all)
                batch_metrics[1].item(),   # FgDice
                batch_metrics[2].item(),   # IoU
                batch_metrics[3].item(),   # HD95
            ))

            if is_main_process() and i < 30:
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                for b in range(imgs.size(0)):
                    save_visualization_4_1(
                        img_tensor=imgs.cpu()[b],
                        mask_tensor=masks.cpu()[b],
                        pred_tensor=preds.cpu()[b],
                        save_dir=vis_save_dir,
                        index=total_saved,
                        num_classes=args.num_classes,
                    )
                    total_saved += 1

    num_top_k_batches = getattr(args, "num_top_k_batches", 5)
    sort_by           = getattr(args, "sort_by", "dice")
    print_top_k_metrics(per_batch_metrics, num_top_k_batches=num_top_k_batches, sort_by=sort_by)

    return run_save_dir
