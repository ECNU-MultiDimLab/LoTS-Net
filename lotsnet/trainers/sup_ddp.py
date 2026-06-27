import os
import torch
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


def train_supervised_multimodal_ddp(
    model,
    train_loader,
    val_loader,
    test_loader,
    train_sampler,
    args,
    single_text_input=False,
):
    device = torch.device(args.gpu)

    scaled_lr = args.lr * (args.world_size if args.world_size > 1 else 1)
    optimizer = optim.AdamW(model.parameters(), lr=scaled_lr, weight_decay=1e-4)


    warmup_ratio = getattr(args, "warmup_ratio", 0.2)
    warmup_epochs = max(1, int(args.epochs * warmup_ratio))
    cosine_epochs = args.epochs - warmup_epochs

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, cosine_epochs),
        eta_min=1e-6,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    criterion = JointSegLoss(
        n_classes=args.num_classes,
        ce_weight=getattr(args, "ce_weight", 1.0),
        dice_weight=getattr(args, "dice_weight", 1.0),
        dice_class_weights=getattr(args, "dice_class_weights", None),
        ce_class_weights=getattr(args, "ce_class_weights", None),
    ).to(device)

    best_dice = 0.0

    best_window_dice = 0.0
    window_start = getattr(args, "best_window_start", 50)
    window_end   = getattr(args, "best_window_end",   100)
    run_save_dir = args.save_dir
    progress = getattr(args, "progress", "tqdm")
    log_interval = getattr(args, "log_interval", 20)


    early_stop_patience = getattr(args, "early_stop_patience", 0)
    early_stop_counter  = 0
    should_stop         = False

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

        train_sampler.set_epoch(epoch)
        model.train()
        train_loss_meter      = AverageMeter()
        train_main_loss_meter = AverageMeter()
        train_main_ce_meter   = AverageMeter()
        train_main_dice_meter = AverageMeter()
        train_aux_loss_meter  = AverageMeter()
        train_aux_ce_meter    = AverageMeter()
        train_aux_dice_meter  = AverageMeter()

        loader_iter = wrap_dataloader(
            train_loader,
            progress=progress,
            desc=f"Epoch {epoch}/{args.epochs} [Train]",
            leave=False,
        )

        for step, batch in enumerate(loader_iter):
            imgs = batch["image"].to(device, non_blocking=True).float()
            masks = batch["mask"].to(device, non_blocking=True).long()

            optimizer.zero_grad()

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                if single_text_input:
                    text_input = (
                        batch["text_spa_spec"].to(device, non_blocking=True).float()
                    )
                    outputs = model(imgs, text_input)
                else:
                    text_spec = batch["text_spec"].to(device, non_blocking=True).float()
                    text_spa = batch["text_spa"].to(device, non_blocking=True).float()
                    outputs = model(imgs, text_spec, text_spa)

                if isinstance(outputs, (tuple, list)):
                    logits = outputs[0]
                    aux_out = outputs[1]
                else:
                    logits = outputs
                    aux_out = None

                loss_main, main_ce, main_dice = criterion(logits, masks, return_components=True)

                if aux_out is not None:
                    if aux_out.dim() > 1:
                        loss_aux, aux_ce, aux_dice = criterion(aux_out, masks, return_components=True)
                    else:
                        loss_aux = aux_out
                        aux_ce = aux_out.detach()
                        aux_dice = aux_out.detach()
                    lambda_aux_sup = getattr(args, "lambda_aux_sup", 0.6)
                    loss = loss_main + lambda_aux_sup * loss_aux
                else:
                    loss = loss_main
                    aux_ce = aux_dice = None

            loss.backward()
            optimizer.step()

            reduced_loss = reduce_tensor(loss.data, args.world_size)
            train_loss_meter.update(reduced_loss.item(), imgs.size(0))

            reduced_main = reduce_tensor(loss_main.data, args.world_size)
            train_main_loss_meter.update(reduced_main.item(), imgs.size(0))
            train_main_ce_meter.update(reduce_tensor(main_ce.data, args.world_size).item(), imgs.size(0))
            train_main_dice_meter.update(reduce_tensor(main_dice.data, args.world_size).item(), imgs.size(0))

            if aux_out is not None:
                reduced_aux = reduce_tensor(loss_aux.data, args.world_size)
                train_aux_loss_meter.update(reduced_aux.item(), imgs.size(0))
                if aux_ce is not None:
                    train_aux_ce_meter.update(reduce_tensor(aux_ce.data, args.world_size).item(), imgs.size(0))
                    train_aux_dice_meter.update(reduce_tensor(aux_dice.data, args.world_size).item(), imgs.size(0))

            maybe_log_step(
                progress=progress,
                step=step,
                total=len(train_loader),
                log_interval=log_interval,
                msg=(
                    f"loss={train_loss_meter.avg:.4f} "
                    f"[main: ce={train_main_ce_meter.avg:.4f}, "
                    f"dice={train_main_dice_meter.avg:.4f}, "
                    f"tot={train_main_loss_meter.avg:.4f} | "
                    f"aux: ce={train_aux_ce_meter.avg:.4f}, "
                    f"dice={train_aux_dice_meter.avg:.4f}, "
                    f"tot={train_aux_loss_meter.avg:.4f}]"
                ),
            )
            maybe_set_postfix(
                loader_iter,
                progress=progress,
                Tot=f"{train_loss_meter.avg:.3f}",
                MCE=f"{train_main_ce_meter.avg:.3f}",
                MD=f"{train_main_dice_meter.avg:.3f}",
                ACE=f"{train_aux_ce_meter.avg:.3f}",
                AD=f"{train_aux_dice_meter.avg:.3f}",
            )

        scheduler.step()


        model.eval()
        val_dice_meter = AverageMeter()
        val_iou_meter  = AverageMeter()
        val_hd95_meter = AverageMeter()

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device, non_blocking=True).float()
                masks = batch["mask"].to(device, non_blocking=True).long()

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    if single_text_input:
                        text_input = (
                            batch["text_spa_spec"].to(device, non_blocking=True).float()
                        )
                        logits = model(imgs, text_input)
                    else:
                        text_spec = batch["text_spec"].to(device, non_blocking=True).float()
                        text_spa = batch["text_spa"].to(device, non_blocking=True).float()
                        logits = model(imgs, text_spec, text_spa)

                    if isinstance(logits, (tuple, list)):
                        logits = logits[0]

                metrics = compute_metrics(
                    logits, masks, args.num_classes,
                    class_weights=getattr(args, "val_metric_class_weights", None),
                )

                metrics_tensor = torch.tensor(
                    [metrics["Dice"], metrics["IoU"], metrics["HD95"]], device=device
                )
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
                metrics_tensor /= args.world_size

                val_dice_meter.update(metrics_tensor[0].item(), imgs.size(0))
                val_iou_meter.update(metrics_tensor[1].item(), imgs.size(0))
                val_hd95_meter.update(metrics_tensor[2].item(), imgs.size(0))

        if is_main_process():
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}: "
                f"Loss: {train_loss_meter.avg:.4f} "
                f"[main: ce={train_main_ce_meter.avg:.4f}, "
                f"dice={train_main_dice_meter.avg:.4f}, "
                f"tot={train_main_loss_meter.avg:.4f} | "
                f"aux: ce={train_aux_ce_meter.avg:.4f}, "
                f"dice={train_aux_dice_meter.avg:.4f}, "
                f"tot={train_aux_loss_meter.avg:.4f}] | "
                f"Val Dice: {val_dice_meter.avg:.4f} | "
                f"Val IoU: {val_iou_meter.avg:.4f} | "
                f"Val HD95: {val_hd95_meter.avg:.4f} | "
                f"LR: {current_lr:.1e}"
            )

            save_dict = {
                "epoch": epoch,
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(save_dict, os.path.join(run_save_dir, "model_latest.pth"))

            if val_dice_meter.avg > best_dice:
                best_dice = val_dice_meter.avg
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

            if window_start <= epoch <= window_end and val_dice_meter.avg > best_window_dice:
                best_window_dice = val_dice_meter.avg
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


    test_model_configs = [
        {
            "path": os.path.join(run_save_dir, "model_best.pth"),
            "label": f"Best Overall",
            "vis_subdir": "test_visual_results",
        },
        {
            "path": os.path.join(run_save_dir, "model_best_window.pth"),
            "label": f"Best Window (ep{window_start}-{window_end})",
            "vis_subdir": f"test_visual_results_window{window_start}_{window_end}",
        },
    ]

    for cfg in test_model_configs:
        model_path = cfg["path"]
        label = cfg["label"]
        if not os.path.exists(model_path):
            master_print(f">>> Warning: {label} model not found at {model_path}, skipping.")
            continue

        master_print(f"\n>>> Testing [{label}] from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.module.load_state_dict(checkpoint["model"])
        saved_epoch = checkpoint.get("epoch", "?")
        model.eval()

        test_dice_meter = AverageMeter()
        test_iou_meter  = AverageMeter()
        test_hd95_meter = AverageMeter()

        vis_save_dir = None
        if is_main_process():
            vis_save_dir = os.path.join(run_save_dir, cfg["vis_subdir"])
            os.makedirs(vis_save_dir, exist_ok=True)

        test_iter = wrap_dataloader(
            test_loader,
            progress=progress,
            desc=f"Testing [{label}]",
            leave=True,
        )

        total_saved = 0
        with torch.no_grad():
            for i, batch in enumerate(test_iter):
                imgs  = batch["image"].to(device, non_blocking=True).float()
                masks = batch["mask"].to(device, non_blocking=True).long()

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    if single_text_input:
                        text_input = batch["text_spa_spec"].to(device, non_blocking=True).float()
                        logits = model(imgs, text_input)
                    else:
                        text_spec = batch["text_spec"].to(device, non_blocking=True).float()
                        text_spa  = batch["text_spa"].to(device, non_blocking=True).float()
                        logits = model(imgs, text_spec, text_spa)

                    if isinstance(logits, (tuple, list)):
                        logits = logits[0]

                metrics = compute_metrics(
                    logits, masks, args.num_classes,
                    class_weights=getattr(args, "test_metric_class_weights", None),
                )

                batch_metrics = torch.tensor(
                    [metrics["Dice"], metrics["IoU"], metrics["HD95"]],
                    device=device,
                )
                dist.all_reduce(batch_metrics, op=dist.ReduceOp.SUM)
                batch_metrics /= args.world_size

                current_batch_size = imgs.size(0) * args.world_size
                test_dice_meter.update(batch_metrics[0].item(), current_batch_size)
                test_iou_meter.update(batch_metrics[1].item(), current_batch_size)
                test_hd95_meter.update(batch_metrics[2].item(), current_batch_size)

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

        n_batches = test_dice_meter.count // (imgs.size(0) * args.world_size)
        master_print("\n" + "=" * 40)
        master_print(f"FINAL TEST RESULTS [{label}] (saved from ep{saved_epoch}):")
        master_print(f"  [All {n_batches:4d} batches]  "
                     f"Dice: {test_dice_meter.avg:.4f}  "
                     f"IoU: {test_iou_meter.avg:.4f}  "
                     f"HD95: {test_hd95_meter.avg:.4f}")
        master_print("=" * 40)

    return run_save_dir
