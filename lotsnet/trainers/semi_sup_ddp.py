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


def get_rampup_weight(current_epoch, T_max, lambda_max):
    if current_epoch >= T_max:
        return lambda_max
    else:
        return lambda_max * math.exp(-5.0 * (1.0 - current_epoch / T_max) ** 2)


def print_top_k_metrics(per_batch_metrics, num_top_k_batches=5, sort_by="dice"):
    if not per_batch_metrics:
        master_print("No test metrics collected.")
        return


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

    best_dice        = 0.0
    best_window_dice = 0.0
    window_start = getattr(args, "best_window_start", 50)
    window_end   = getattr(args, "best_window_end",   100)
    early_stop_patience = getattr(args, "early_stop_patience", 0)
    early_stop_counter  = 0
    should_stop         = False
    run_save_dir = args.save_dir
    progress = getattr(args, "progress", "tqdm")
    log_interval = getattr(args, "log_interval", 20)
    if is_main_process():
        os.makedirs(run_save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        if early_stop_patience > 0:
            stop_flag = torch.tensor(int(should_stop), dtype=torch.int32, device=device)
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item():
                master_print(
                    f"\n>>> Early stopping triggered at epoch {epoch - 1}. "
                    f"No improvement for {early_stop_patience} consecutive epochs. "
                    f"Best Val Dice: {best_dice:.4f}"
                )
                break

        labeled_sampler.set_epoch(epoch)
        unlabeled_sampler.set_epoch(epoch)

        model.train()
        loss_meter          = AverageMeter()
        sup_loss_meter      = AverageMeter()
        sup_main_meter      = AverageMeter()
        sup_main_ce_meter   = AverageMeter()
        sup_main_dice_meter = AverageMeter()
        sup_aux_meter       = AverageMeter()
        sup_aux_ce_meter    = AverageMeter()
        sup_aux_dice_meter  = AverageMeter()
        unsup_loss_meter    = AverageMeter()

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


            img_l = batch_l["image"].to(device, non_blocking=True).float()
            mask_l = batch_l["mask"].to(device, non_blocking=True).long()
            text_spec_l = batch_l["text_spec"].to(device, non_blocking=True).float()
            text_spa_l = batch_l["text_spa"].to(device, non_blocking=True).float()

            with model.no_sync():
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    out_l, aux_l = model(img_l, text_spec_l, text_spa_l, input_data_type=0)
                    loss_sup_main, sup_main_ce, sup_main_dice = criterion_seg(
                        out_l, mask_l, return_components=True
                    )
                    if aux_l is not None:
                        loss_sup_aux_raw, sup_aux_ce, sup_aux_dice = criterion_seg(
                            aux_l, mask_l, return_components=True
                        )
                    else:
                        loss_sup_aux_raw = torch.zeros(1, device=device)
                        sup_aux_ce = sup_aux_dice = loss_sup_aux_raw
                    loss_sup = loss_sup_main + args.lambda_aux * loss_sup_aux_raw
                loss_sup.backward()

            sup_loss_val = loss_sup.item()
            total_loss_val += sup_loss_val
            sup_main_meter.update(loss_sup_main.item())
            sup_main_ce_meter.update(sup_main_ce.item())
            sup_main_dice_meter.update(sup_main_dice.item())
            sup_aux_meter.update(loss_sup_aux_raw.item())
            sup_aux_ce_meter.update(sup_aux_ce.item())
            sup_aux_dice_meter.update(sup_aux_dice.item())
            del img_l, mask_l, text_spec_l, text_spa_l, out_l, aux_l
            del loss_sup, loss_sup_main, loss_sup_aux_raw
            del sup_main_ce, sup_main_dice, sup_aux_ce, sup_aux_dice


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


            optimizer.step()

            loss_meter.update(total_loss_val, batch_l["image"].size(0))
            sup_loss_meter.update(sup_loss_val)
            if args.use_l_pl or args.use_l_con:
                unsup_loss_meter.update(unsup_loss_val)

            step_msg = (
                f"Tot={loss_meter.avg:.3f}  "
                f"S={sup_loss_meter.avg:.3f}"
                f"(ce={sup_main_ce_meter.avg:.3f},dice={sup_main_dice_meter.avg:.3f},"
                f"aux_ce={sup_aux_ce_meter.avg:.3f},aux_dice={sup_aux_dice_meter.avg:.3f})  "
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
                MCE=f"{sup_main_ce_meter.avg:.3f}",
                MD=f"{sup_main_dice_meter.avg:.3f}",
                ACE=f"{sup_aux_ce_meter.avg:.3f}",
                AD=f"{sup_aux_dice_meter.avg:.3f}",
                U=f"{unsup_loss_meter.avg:.3f}",
                W=f"{rampup_weight:.2f}",
            )

        scheduler.step()


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
                    class_weights=getattr(args, "val_metric_class_weights", None),
                )
                metrics_tensor = torch.tensor(
                    [metrics["Dice"], metrics["FgDice"],
                     metrics["IoU"], metrics["HD95"]], device=device
                )
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
                metrics_tensor /= args.world_size

                val_batch_metrics.append((
                    metrics_tensor[0].item(),
                    metrics_tensor[1].item(),
                    metrics_tensor[2].item(),
                    metrics_tensor[3].item(),
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
                f"[main: ce={sup_main_ce_meter.avg:.4f}, dice={sup_main_dice_meter.avg:.4f} | "
                f"aux: ce={sup_aux_ce_meter.avg:.4f}, dice={sup_aux_dice_meter.avg:.4f}] | "
                f"Val Dice: {val_dice_avg:.4f} | "
                f"Val IoU: {val_iou_avg:.4f} | "
                f"Val HD95: {val_hd95_avg:.4f} | "
                f"LR: {current_lr:.1e}"
            )

            save_dict = {
                "epoch": epoch,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(save_dict, os.path.join(run_save_dir, "model_latest.pth"))

            if val_dice_avg > best_dice:
                best_dice = val_dice_avg
                torch.save(save_dict, os.path.join(run_save_dir, "model_best.pth"))
                print(f">>> New Best Model Saved! Dice: {best_dice:.4f}")
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_patience > 0:
                    print(
                        f"[EarlyStop] No improvement for {early_stop_counter}/"
                        f"{early_stop_patience} epochs. Best Dice: {best_dice:.4f}"
                    )

            if window_start <= epoch <= window_end and val_dice_avg > best_window_dice:
                best_window_dice = val_dice_avg
                torch.save(save_dict, os.path.join(run_save_dir, "model_best_window.pth"))
                print(
                    f">>> New Best Window Model Saved! "
                    f"(ep{window_start}-{window_end}) Dice: {best_window_dice:.4f}"
                )

            if early_stop_patience > 0 and early_stop_counter >= early_stop_patience:
                should_stop = True


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
                class_weights=getattr(args, "test_metric_class_weights", None),
            )
            batch_metrics = torch.tensor(
                [metrics["Dice"], metrics["FgDice"],
                 metrics["IoU"], metrics["HD95"]], device=device
            )
            dist.all_reduce(batch_metrics, op=dist.ReduceOp.SUM)
            batch_metrics /= args.world_size

            per_batch_metrics.append((
                batch_metrics[0].item(),
                batch_metrics[1].item(),
                batch_metrics[2].item(),
                batch_metrics[3].item(),
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
