import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from monai.metrics import compute_hausdorff_distance


class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes, smooth=1e-5, class_weights=None):
        super(SoftDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.smooth = smooth
        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", w / w.sum())
        else:
            self.class_weights = None

    def forward(self, logits, target):

        probs = F.softmax(logits, dim=1)


        target_onehot = (
            F.one_hot(target, num_classes=self.n_classes).permute(0, 3, 1, 2).float()
        )


        dims = (2, 3)
        intersection = torch.sum(probs * target_onehot, dim=dims)
        cardinality = torch.sum(probs + target_onehot, dim=dims)


        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)


        dice_per_class = dice_score.mean(dim=0)


        if self.class_weights is not None:
            return 1.0 - (dice_per_class * self.class_weights.to(dice_per_class.device)).sum()
        else:
            return 1.0 - dice_per_class.mean()


class JointSegLoss(nn.Module):

    def __init__(
        self,
        n_classes,
        ce_weight=1.0,
        dice_weight=1.0,
        dice_class_weights=None,
        ce_class_weights=None,
    ):
        super(JointSegLoss, self).__init__()

        if ce_class_weights is not None:
            ce_w = torch.tensor(ce_class_weights, dtype=torch.float32)
        else:
            ce_w = torch.tensor([1.0, 1.0], dtype=torch.float32)

        self.register_buffer("ce_class_weights_buf", ce_w)


        self._n_classes = n_classes
        self.dice = SoftDiceLoss(n_classes, class_weights=dice_class_weights)
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target, return_components=False):

        ce = nn.functional.cross_entropy(
            logits, target,
            weight=self.ce_class_weights_buf.to(logits.device),
        )
        loss_ce   = ce
        loss_dice = self.dice(logits, target)
        total     = self.ce_weight * loss_ce + self.dice_weight * loss_dice

        if return_components:
            return total, loss_ce, loss_dice
        return total


def compute_metrics(logits, target, n_classes, class_weights=None):


    probs = F.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)


    dice_list = []
    iou_list = []


    for c in range(n_classes):
        pred_c = preds == c
        target_c = target == c

        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        target_area = target_c.sum().float()
        pred_area = pred_c.sum().float()


        dice = (2 * intersection) / (target_area + pred_area + 1e-5)


        iou = intersection / (union + 1e-5)

        dice_list.append(dice)
        iou_list.append(iou)


    dice_tensor = torch.stack(dice_list)
    iou_tensor  = torch.stack(iou_list)

    if class_weights is not None and len(class_weights) == n_classes:


        w = torch.tensor(
            list(reversed(class_weights)), dtype=torch.float32, device=dice_tensor.device
        )
        w = w / w.sum()
        mean_dice = (dice_tensor * w).sum().item()
        mean_iou  = (iou_tensor  * w).sum().item()
    else:
        mean_dice = dice_tensor.mean().item()
        mean_iou  = iou_tensor.mean().item()


    fg_dice = dice_tensor[1:].mean().item() if len(dice_list) > 1 else mean_dice


    y_pred_onehot = F.one_hot(preds, num_classes=n_classes).permute(0, 3, 1, 2).float()
    y_onehot = F.one_hot(target, num_classes=n_classes).permute(0, 3, 1, 2).float()

    try:


        hd95_per_class_per_batch = compute_hausdorff_distance(
            y_pred_onehot, y_onehot, percentile=95, include_background=False
        )


        hd95_per_class_per_batch[torch.isinf(hd95_per_class_per_batch)] = torch.nan


        mean_hd95 = torch.nanmean(hd95_per_class_per_batch).item()

    except Exception as e:

        print(e)
        mean_hd95 = 100.0
    return {"Dice": mean_dice, "FgDice": fg_dice, "IoU": mean_iou, "HD95": mean_hd95}


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_visualization(
    img_tensor, mask_tensor, pred_tensor, save_dir, index, num_classes=2
):

    img = img_tensor.permute(1, 2, 0).numpy()


    if img.shape[2] > 3:
        img_display = np.mean(img, axis=2)
    elif img.shape[2] == 1:
        img_display = img.squeeze(2)
    else:
        img_display = img


    min_val, max_val = np.min(img_display), np.max(img_display)
    if max_val > min_val:
        img_display = (img_display - min_val) / (max_val - min_val)


    mask = mask_tensor.numpy()
    pred = pred_tensor.numpy()


    plt.figure(figsize=(12, 10))
    plt.subplots_adjust(
        left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1
    )


    plt.subplot(2, 2, 1)
    plt.imshow(img_display, cmap="gray" if len(img_display.shape) == 2 else None)
    plt.title("Original (Avg Channel)")
    plt.axis("off")


    plt.subplot(2, 2, 2)
    plt.imshow(mask, cmap="gray", vmin=0, vmax=num_classes - 1)
    plt.title("Ground Truth")
    plt.axis("off")


    plt.subplot(2, 2, 3)
    plt.imshow(pred, cmap="gray", vmin=0, vmax=num_classes - 1)
    plt.title("Prediction")
    plt.axis("off")


    plt.subplot(2, 2, 4)
    plt.imshow(img_display, cmap="gray" if len(img_display.shape) == 2 else None)

    overlay = np.zeros((*pred.shape, 4))
    overlay[pred == 1] = [1, 0, 0, 0.4]
    if num_classes > 2:

        overlay[pred == 2] = [0, 1, 0, 0.4]

    plt.imshow(overlay)
    plt.title("Prediction Overlay")
    plt.axis("off")


    save_path = os.path.join(save_dir, f"sample_{index}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close()


def save_visualization_4_1(
    img_tensor, mask_tensor, pred_tensor, save_dir, index, num_classes=2
):

    img = img_tensor.permute(1, 2, 0).numpy()


    if img.shape[2] > 3:
        img_display = np.mean(img, axis=2)
    elif img.shape[2] == 1:
        img_display = img.squeeze(2)
    else:
        img_display = img


    min_val, max_val = np.min(img_display), np.max(img_display)
    if max_val > min_val:
        img_display = (img_display - min_val) / (max_val - min_val)
    else:
        img_display = np.zeros_like(img_display)


    mask = mask_tensor.numpy()
    pred = pred_tensor.numpy()


    overlay = np.zeros((*pred.shape, 4))
    overlay[pred == 1] = [1, 0, 0, 0.4]
    if num_classes > 2:
        overlay[pred == 2] = [0, 1, 0, 0.4]


    plt.figure(figsize=(6, 24))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.2)


    cmap_mode = "gray" if len(img_display.shape) == 2 else None


    plt.subplot(4, 1, 1)
    plt.imshow(img_display, cmap=cmap_mode)
    plt.title("Original Image")
    plt.axis("off")


    plt.subplot(4, 1, 2)
    plt.imshow(img_display, cmap=cmap_mode)
    plt.imshow(overlay)
    plt.title("Prediction Overlay")
    plt.axis("off")


    plt.subplot(4, 1, 3)
    plt.imshow(pred, cmap="gray", vmin=0, vmax=num_classes - 1)
    plt.title("Prediction Mask")
    plt.axis("off")


    plt.subplot(4, 1, 4)
    plt.imshow(mask, cmap="gray", vmin=0, vmax=num_classes - 1)
    plt.title("Ground Truth")
    plt.axis("off")


    save_path = os.path.join(save_dir, f"sample_{index}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close()


def save_evolution_visualization(
    dataset, history_dict, save_dir, num_classes=2, max_samples=50
):

    vis_dir = os.path.join(save_dir, "pseudo_label_evolution")
    os.makedirs(vis_dir, exist_ok=True)


    recorded_epochs = sorted(history_dict.keys())
    num_snapshots = len(recorded_epochs)

    print(
        f"\n>>> Generating Evolution Visualization for {min(len(dataset), max_samples)} samples..."
    )
    print(f"    Snapshots at epochs: {recorded_epochs}")


    for idx in tqdm(range(min(len(dataset), max_samples)), desc="Saving Evolution"):

        img_tensor, mask_tensor = dataset[idx]

        mask = mask_tensor.squeeze().numpy()


        cols = 1 + num_snapshots
        fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3))


        plt.subplots_adjust(wspace=0.1, hspace=0)


        ax_gt = axes[0]
        ax_gt.imshow(mask, cmap="gray", vmin=0, vmax=num_classes - 1)
        ax_gt.set_title("Ground Truth", fontsize=10)
        ax_gt.axis("off")


        for i, epoch in enumerate(recorded_epochs):
            ax = axes[i + 1]


            pred_mask = history_dict[epoch][idx]

            ax.imshow(pred_mask, cmap="gray", vmin=0, vmax=num_classes - 1)
            ax.set_title(f"Ep {epoch}", fontsize=10)
            ax.axis("off")


        save_path = os.path.join(vis_dir, f"sample_{idx}_evolution.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=100)
        plt.close(fig)

    print(f">>> Evolution visualization saved to {vis_dir}")
