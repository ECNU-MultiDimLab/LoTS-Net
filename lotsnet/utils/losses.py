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
        """
        多分类 Soft Dice Loss

        Args:
            n_classes:     类别数
            smooth:        平滑项，防止分母为零
            class_weights: 每个类别在 Dice 均值中的权重，长度须等于 n_classes。
                           None        → 所有类别等权（原始行为）
                           [0.0, 1.0]  → 纯前景 Dice（完全忽略背景）
                           [0.5, 1.0]  → 前景权重是背景的 2 倍
                           权重在内部会 L1 归一化（sum=1），无需手动归一化。
        """
        super(SoftDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.smooth = smooth
        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", w / w.sum())
        else:
            self.class_weights = None

    def forward(self, logits, target):
        """
        Args:
            logits: [B, C, H, W] (模型输出的原始 Logits)
            target: [B, H, W] (标签，值为 0 ~ C-1)
        """
        # 1. Apply Softmax
        probs = F.softmax(logits, dim=1)

        # 2. One-hot Encoding for Target
        target_onehot = (
            F.one_hot(target, num_classes=self.n_classes).permute(0, 3, 1, 2).float()
        )

        # 3. Calculate Intersection and Cardinality per (Batch, Class)
        dims = (2, 3)
        intersection = torch.sum(probs * target_onehot, dim=dims)
        cardinality = torch.sum(probs + target_onehot, dim=dims)

        # 4. Dice Score: [B, C]
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # 5. 先在 Batch 维度平均得到每类 Dice: [C]
        dice_per_class = dice_score.mean(dim=0)

        # 6. 按类别权重加权平均（或等权）
        if self.class_weights is not None:
            return 1.0 - (dice_per_class * self.class_weights.to(dice_per_class.device)).sum()
        else:
            return 1.0 - dice_per_class.mean()


class JointSegLoss(nn.Module):
    """
    组合损失: L_seg = CrossEntropy + SoftDice
    """

    def __init__(
        self,
        n_classes,
        ce_weight=1.0,
        dice_weight=1.0,
        dice_class_weights=None,
        ce_class_weights=None,
    ):
        """
        Args:
            ce_weight:          CE loss 整体系数（相对于 Dice 的比例），默认 1.0。
            dice_weight:        Dice loss 整体系数，默认 1.0。
            dice_class_weights: 透传给 SoftDiceLoss 的类别权重列表，None 表示等权。
            ce_class_weights:   CrossEntropyLoss 的类别权重列表。
                                None → 等权 [1.0, 1.0]（默认，语义透明）。
                                示例：[1.0, 6.0] 表示前景权重为背景的 6 倍（逆频率加权）。
                                权重将在 forward 时动态移到正确设备，无需手动指定 .cuda()。
        """
        super(JointSegLoss, self).__init__()

        if ce_class_weights is not None:
            ce_w = torch.tensor(ce_class_weights, dtype=torch.float32)
        else:
            ce_w = torch.tensor([1.0, 1.0], dtype=torch.float32)
        # register_buffer 使权重随模型 .to(device) 自动迁移，避免 .cuda() 硬编码
        self.register_buffer("ce_class_weights_buf", ce_w)

        # CE loss 在 forward 时用动态权重，先占位 None
        self._n_classes = n_classes
        self.dice = SoftDiceLoss(n_classes, class_weights=dice_class_weights)
        self.ce_weight   = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target, return_components=False):
        """
        Args:
            return_components: 若为 True，返回 (total, ce_loss, dice_loss)；
                               否则仅返回 total（默认，向后兼容）。
        """
        # CE 权重动态适配当前设备（兼容多卡 DDP 各 rank 不同 device）
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


# =========================================================
# 2. 评估指标 (Evaluation Metrics)
# =========================================================


def compute_metrics(logits, target, n_classes, class_weights=None):
    """
    计算验证集的一个 Batch 的 Dice 和 IoU

    Args:
        logits:         [B, C, H, W]
        target:         [B, H, W]
        class_weights:  可选，长度为 n_classes 的列表/元组，用于 Dice/IoU 的加权平均。
                        约定：第 0 个元素为前景权重，第 1 个元素为背景权重。
                        例如 [1.0, 1.8] 表示前景权重 1.0、背景权重 1.8，
                        适合背景像素约为前景 1.8 倍的场景（像素占比加权）。
                        None → 等权平均（默认）。
    Returns:
        metrics: dict {'Dice': float, 'FgDice': float, 'IoU': float, 'HD95': float}
    """
    # 1. 获取预测类别
    # [B, C, H, W] -> [B, H, W]
    probs = F.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

    # 转换为 CPU numpy 方便计算 (如果显存够大也可以在 GPU 上算)
    # 这里保持在 GPU 上计算以提高速度

    dice_list = []
    iou_list = []

    # 2. 逐类别计算 (忽略背景类通常是可选的，这里默认计算所有类别然后平均)
    # 如果你想忽略背景 (class 0)，将 range 改为 range(1, n_classes)
    for c in range(n_classes):
        pred_c = preds == c
        target_c = target == c

        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        target_area = target_c.sum().float()
        pred_area = pred_c.sum().float()

        # Dice = 2 * Inter / (Area_Pred + Area_Target)
        dice = (2 * intersection) / (target_area + pred_area + 1e-5)

        # IoU = Inter / Union
        iou = intersection / (union + 1e-5)

        dice_list.append(dice)
        iou_list.append(iou)

    # 3. 计算 Mean Metrics
    dice_tensor = torch.stack(dice_list)
    iou_tensor  = torch.stack(iou_list)

    if class_weights is not None and len(class_weights) == n_classes:
        # class_weights 约定：[0]=前景权重, [1]=背景权重
        # dice_list 顺序：[0]=背景(class 0), [1]=前景(class 1)
        # 因此需要把权重反转后对应到 dice_list 的顺序：
        #   背景 dice 使用 class_weights[1]，前景 dice 使用 class_weights[0]
        w = torch.tensor(
            list(reversed(class_weights)), dtype=torch.float32, device=dice_tensor.device
        )
        w = w / w.sum()
        mean_dice = (dice_tensor * w).sum().item()
        mean_iou  = (iou_tensor  * w).sum().item()
    else:
        mean_dice = dice_tensor.mean().item()
        mean_iou  = iou_tensor.mean().item()

    # 前景 Dice：跳过 class 0（背景），对 class 1..n_classes-1 求均值
    fg_dice = dice_tensor[1:].mean().item() if len(dice_list) > 1 else mean_dice

    # 1. 将预测和标签转为 One-Hot 格式 [B, C, H, W]
    # 假设 preds 是 [B, H, W], target 是 [B, H, W]
    # print(preds.shape, target.shape)
    # print(preds.unique(), target.unique())
    y_pred_onehot = F.one_hot(preds, num_classes=n_classes).permute(0, 3, 1, 2).float()
    y_onehot = F.one_hot(target, num_classes=n_classes).permute(0, 3, 1, 2).float()

    try:
        # compute_hausdorff_distance 会返回 [Batch, n_classes] 的矩阵
        # 每行代表一个样本，每列代表一个类别的 HD95 值
        hd95_per_class_per_batch = compute_hausdorff_distance(
            y_pred_onehot, y_onehot, percentile=95, include_background=False
        )

        # include_background=False 会自动跳过 Class 0
        # 此时 hd95_per_class_per_batch 的列数是 n_classes - 1

        # 计算所有前景类的平均值
        # 注意：如果某个样本中完全没预测出某个类，MONAI 会返回 inf 或 NaN
        # 我们需要过滤掉这些无效值再算平均数
        hd95_per_class_per_batch[torch.isinf(hd95_per_class_per_batch)] = torch.nan

        # 先在类之间平均，再在 Batch 之间平均
        mean_hd95 = torch.nanmean(hd95_per_class_per_batch).item()

    except Exception as e:
        # 如果整个 Batch 都没分出任何目标，或者计算出错
        print(e)
        mean_hd95 = 100.0  # 或者使用 float('nan')
    return {"Dice": mean_dice, "FgDice": fg_dice, "IoU": mean_iou, "HD95": mean_hd95}


# =========================================================
# 3. 辅助工具 (AverageMeter)
# =========================================================


class AverageMeter(object):
    """
    计算并存储平均值和当前值，用于 Log 记录
    """

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
    """
    保存可视化结果：原图(Hyperspectral avg)、真值、预测值、叠加图

    Args:
        img_tensor: [C, H, W] 原始图像张量 (CPU)
        mask_tensor: [H, W] 真值掩码张量 (CPU)
        pred_tensor: [H, W] 预测掩码张量 (CPU, 已经是 argmax 后的结果)
        save_dir: 保存目录
        index: 文件名索引
    """
    # 1. 处理图像 (C, H, W) -> (H, W, C)
    img = img_tensor.permute(1, 2, 0).numpy()

    # 高光谱处理：如果通道 > 3，计算平均值变成灰度图显示
    if img.shape[2] > 3:
        img_display = np.mean(img, axis=2)
    elif img.shape[2] == 1:
        img_display = img.squeeze(2)
    else:
        img_display = img  # RGB

    # 归一化到 [0, 1] 用于显示
    min_val, max_val = np.min(img_display), np.max(img_display)
    if max_val > min_val:
        img_display = (img_display - min_val) / (max_val - min_val)

    # 2. 处理 Mask 和 Pred
    mask = mask_tensor.numpy()
    pred = pred_tensor.numpy()

    # 3. 绘图
    plt.figure(figsize=(12, 10))
    plt.subplots_adjust(
        left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1
    )

    # Subplot 1: Original Image
    plt.subplot(2, 2, 1)
    plt.imshow(img_display, cmap="gray" if len(img_display.shape) == 2 else None)
    plt.title("Original (Avg Channel)")
    plt.axis("off")

    # Subplot 2: Ground Truth
    plt.subplot(2, 2, 2)
    plt.imshow(mask, cmap="gray", vmin=0, vmax=num_classes - 1)
    plt.title("Ground Truth")
    plt.axis("off")

    # Subplot 3: Prediction
    plt.subplot(2, 2, 3)
    plt.imshow(pred, cmap="gray", vmin=0, vmax=num_classes - 1)
    plt.title("Prediction")
    plt.axis("off")

    # Subplot 4: Overlay (Image + Pred)
    plt.subplot(2, 2, 4)
    plt.imshow(img_display, cmap="gray" if len(img_display.shape) == 2 else None)
    # 创建一个带颜色的 mask (例如红色表示病灶)
    # 假设 class 1 是病灶
    overlay = np.zeros((*pred.shape, 4))  # RGBA
    overlay[pred == 1] = [1, 0, 0, 0.4]  # Red with alpha 0.4
    if num_classes > 2:
        # 如果是多分类，可以加其他颜色
        overlay[pred == 2] = [0, 1, 0, 0.4]  # Green

    plt.imshow(overlay)
    plt.title("Prediction Overlay")
    plt.axis("off")

    # 保存
    save_path = os.path.join(save_dir, f"sample_{index}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close()


def save_visualization_4_1(
    img_tensor, mask_tensor, pred_tensor, save_dir, index, num_classes=2
):
    """
    保存可视化结果：4x1 排列
    顺序：原图 -> 预测叠加图 -> 预测Mask -> 真值Mask

    Args:
        img_tensor: [C, H, W] 原始图像张量 (CPU)
        mask_tensor: [H, W] 真值掩码张量 (CPU)
        pred_tensor: [H, W] 预测掩码张量 (CPU, 已经是 argmax 后的结果)
        save_dir: 保存目录
        index: 文件名索引
    """
    # 1. 处理图像 (C, H, W) -> (H, W, C)
    img = img_tensor.permute(1, 2, 0).numpy()

    # 高光谱处理：如果通道 > 3，计算平均值变成灰度图显示
    if img.shape[2] > 3:
        img_display = np.mean(img, axis=2)
    elif img.shape[2] == 1:
        img_display = img.squeeze(2)
    else:
        img_display = img  # RGB

    # 归一化到 [0, 1] 用于显示
    min_val, max_val = np.min(img_display), np.max(img_display)
    if max_val > min_val:
        img_display = (img_display - min_val) / (max_val - min_val)
    else:
        img_display = np.zeros_like(img_display)

    # 2. 处理 Mask 和 Pred
    mask = mask_tensor.numpy()
    pred = pred_tensor.numpy()

    # 3. 准备叠加层数据 (Overlay Data)
    # 创建一个带颜色的 mask (例如红色表示病灶)
    # 假设 class 1 是病灶 (Red), class 2 是其他 (Green)
    overlay = np.zeros((*pred.shape, 4))  # RGBA
    overlay[pred == 1] = [1, 0, 0, 0.4]  # Red with alpha 0.4
    if num_classes > 2:
        overlay[pred == 2] = [0, 1, 0, 0.4]  # Green with alpha 0.4

    # 4. 绘图设置
    # 因为是 4行1列，高度需要设得比较大，保持长宽比
    plt.figure(figsize=(6, 24))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.2)

    # 灰度图模式判断
    cmap_mode = "gray" if len(img_display.shape) == 2 else None

    # --- Subplot 1: Original Image (原图) ---
    plt.subplot(4, 1, 1)
    plt.imshow(img_display, cmap=cmap_mode)
    plt.title("Original Image")
    plt.axis("off")

    # --- Subplot 2: Prediction Overlay (预测叠加图) ---
    plt.subplot(4, 1, 2)
    plt.imshow(img_display, cmap=cmap_mode)  # 先画底图
    plt.imshow(overlay)  # 再叠加上层
    plt.title("Prediction Overlay")
    plt.axis("off")

    # --- Subplot 3: Prediction Mask (预测值) ---
    plt.subplot(4, 1, 3)
    plt.imshow(pred, cmap="gray", vmin=0, vmax=num_classes - 1)
    plt.title("Prediction Mask")
    plt.axis("off")

    # --- Subplot 4: Ground Truth (真值) ---
    plt.subplot(4, 1, 4)
    plt.imshow(mask, cmap="gray", vmin=0, vmax=num_classes - 1)
    plt.title("Ground Truth")
    plt.axis("off")

    # 保存
    save_path = os.path.join(save_dir, f"sample_{index}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.close()


def save_evolution_visualization(
    dataset, history_dict, save_dir, num_classes=2, max_samples=50
):
    """
    绘制伪标签演变过程图 (1 x (1 + N_snapshots))

    Args:
        dataset: Unlabeled Dataset (现在包含了 Mask)
        history_dict: 字典 {epoch_num: preds_array}
                      preds_array shape: [N_samples, H, W]
        save_dir: 保存目录
        num_classes: 类别数
        max_samples: 限制保存的样本数量，防止生成太多图片
    """
    # 1. 准备目录
    vis_dir = os.path.join(save_dir, "pseudo_label_evolution")
    os.makedirs(vis_dir, exist_ok=True)

    # 2. 获取记录的 Epoch 列表并排序
    recorded_epochs = sorted(history_dict.keys())
    num_snapshots = len(recorded_epochs)

    print(
        f"\n>>> Generating Evolution Visualization for {min(len(dataset), max_samples)} samples..."
    )
    print(f"    Snapshots at epochs: {recorded_epochs}")

    # 3. 遍历数据集
    for idx in tqdm(range(min(len(dataset), max_samples)), desc="Saving Evolution"):
        # 获取 GT Mask 和 原图 (dataset 返回 img, msk)
        # 注意：dataset[idx] 返回的是 tensor，需要转 numpy
        img_tensor, mask_tensor = dataset[idx]

        mask = mask_tensor.squeeze().numpy()  # [H, W]

        # 4. 开始绘图: 1行, (1 + num_snapshots) 列
        # 1 (GT) + N (Pseudo Labels)
        cols = 1 + num_snapshots
        fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3))

        # 调整间距
        plt.subplots_adjust(wspace=0.1, hspace=0)

        # --- 第一张：Ground Truth ---
        ax_gt = axes[0]
        ax_gt.imshow(mask, cmap="gray", vmin=0, vmax=num_classes - 1)
        ax_gt.set_title("Ground Truth", fontsize=10)
        ax_gt.axis("off")

        # --- 后续几张：Pseudo Label Snapshots ---
        for i, epoch in enumerate(recorded_epochs):
            ax = axes[i + 1]

            # 获取该 epoch 下，第 idx 个样本的预测
            # history_dict[epoch] 是一个巨大的 numpy array [N, H, W]
            pred_mask = history_dict[epoch][idx]

            ax.imshow(pred_mask, cmap="gray", vmin=0, vmax=num_classes - 1)
            ax.set_title(f"Ep {epoch}", fontsize=10)
            ax.axis("off")

        # 5. 保存
        save_path = os.path.join(vis_dir, f"sample_{idx}_evolution.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=100)
        plt.close(fig)

    print(f">>> Evolution visualization saved to {vis_dir}")
