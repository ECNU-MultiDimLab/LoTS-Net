#!/bin/bash

# ============================================================
# LoTS-Net 全监督训练脚本
# ============================================================

export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2
MASTER_PORT=$(python3 -c "
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print(s.getsockname()[1])
")
echo "[INFO] Using master port: $MASTER_PORT"

SEED=42
NUM_WORKERS=1

# ---- 数据 ----
DATA_ROOT="data/MDC_Resized_256_320_to_256_256_overlap_0_0"          # 请替换为实际数据路径
VAL_RATIO=0.1
TEST_RATIO=0.1

# ---- 数据集划分模式 ----
# ""                        → 随机划分（默认），按 VAL_RATIO/TEST_RATIO 比例随机分
# "splits/split_sup.csv"    → CSV 预设划分（推荐），由 dataview.ipynb 生成
#                             CSV 需含 'case_id' 和 'split'（train/val/test）列
#                             设置后，VAL_RATIO/TEST_RATIO 参数被忽略
# SPLIT_CSV=""
SPLIT_CSV="data/MDC_Resized_256_320_to_256_256_overlap_0_0/split_sup.csv"

# ---- 训练超参数 ----
BATCH_SIZE=2
EPOCHS=150
LR=0.0004
NUM_CLASSES=2

# ---- LR Schedule ----
# warmup_ratio: 线性 warmup 占总 epoch 的比例（0.2 = 前 20% epoch 用于 warmup）
# 剩余 (1-warmup_ratio) 比例的 epoch 使用 Cosine Annealing 衰减至 eta_min=1e-6
WARMUP_RATIO=0.2

# ---- 早停（Early Stopping）----
# 监测 Val Dice(all)：连续若干 epoch 无改善则中断训练，直接进入测试阶段
# 0  → 禁用早停（默认，走完全部 EPOCHS）
# 20 → 连续 20 epoch 未刷新最佳 Val Dice 则停止（推荐起点）
EARLY_STOP_PATIENCE=0

# ---- 窗口最佳模型保存 ----
# 在 [BEST_WINDOW_START, BEST_WINDOW_END] 区间内额外保存一个最佳验证模型
# 用于对比全程最佳模型，评估训练后期是否存在过拟合
BEST_WINDOW_START=50
BEST_WINDOW_END=100

# ---- 模型超参数 ----
IN_CHANS=60
TEXT_DIM=1024
C_SPE=144
C_ATTN=144
ROUTER_TOP_K=15
STEM_CH=128
QUEUE_LEN=300
QUEUE_DEVICE="cuda"         # "cpu" 或 "cuda"（cuda 可减少 CPU→GPU 传输延迟）

# ---- 队列检索模式 ----
# "top1"    : 论文 Eq.1，全局池化后 argmax 选最匹配条目，显存极低（推荐）
# "chunked" : Online-Softmax 分块密集注意力，与原版数值等价但显存可控
RETRIEVAL_MODE="chunked"
QUEUE_CHUNK_SIZE=32         # 仅 chunked 模式有效：每块处理的队列条目数

# ---- Router Temperature ----
# scoring_mlp 输出的 Top-K logit 在 softmax 前除以此温度
# < 1.0 → 权重更尖锐（接近 hard selection，梯度更集中）
# > 1.0 → 权重更均匀（soft aggregation，训练初期探索更充分）
# 1.0   → 无缩放（默认）
ROUTER_TEMPERATURE=1.0

# ---- SMD EMA 基矩阵 ----
# "--smd_ema_bases"  → 开启（强烈推荐）：NMF 基矩阵持久化+EMA更新
#                     修复梯度检查点+rand_init 导致的错误梯度问题
#                     同时避免 BatchNorm/GroupNorm running stats 每步随机扰动
# ""                → 关闭：每次 forward 随机初始化 NMF 基（原始行为，不推荐）
SMD_EMA_BASES="--smd_ema_bases"

# ---- DDP 选项 ----
# 设为 "--find_unused_parameters" 开启（适合消融实验有未使用参数时）
# 设为 ""                        关闭（全模型训练时关闭可降低通信开销）
FIND_UNUSED_PARAMS=""

# ---- Dice Loss 类别权重 ----
# ""                                  → 不传参数，背景/前景等权（默认行为）
# "--dice_class_weights 0.0 1.0"      → 纯前景 Dice（完全忽略背景）
# "--dice_class_weights 0.3 1.0"      → 前景权重约为背景的 3.3 倍
# "--dice_class_weights 0.5 1.0"      → 前景权重为背景的 2 倍（推荐起点）
DICE_CLASS_WEIGHTS="--dice_class_weights 0.5 1.0"

# ---- CE Loss 类别权重 ----
# compute_metrics 的加权约定：第 0 个值=前景权重，第 1 个值=背景权重
# 背景:前景 ≈ 1.8:1 时，设 [1.0, 1.8] → 背景 dice 权重 0.643，前景 0.357
# 这样 Val Dice(all) ≈ 0.643*bg_dice + 0.357*fg_dice，比等权结果更乐观
# ""                                      → 不传参数，使用等权平均（默认）
# "--ce_class_weights 1.0 1.8"            → 前景权重 1.0，背景权重 1.8（推荐）
CE_CLASS_WEIGHTS=""

# ---- CE/Dice 整体系数 ----
# 控制 CE 和 Dice 在总损失中的相对比例：loss = ce_weight*CE + dice_weight*Dice
CE_WEIGHT=1.0
DICE_WEIGHT=1.0

# ---- 辅助头损失权重 ----
# loss_total = loss_main + LAMBDA_AUX_SUP * loss_aux
# 设为 0.0 可完全禁用辅助头损失（相当于消融辅助头）
LAMBDA_AUX_SUP=0.4

# ---- 日志 ----
PROGRESS="log"              # tqdm / log / none
LOG_INTERVAL=20
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
EXP_NAME="Sup_LoTSNet_${TIMESTAMP}"
SAVE_DIR="./records/${EXP_NAME}"
mkdir -p "$SAVE_DIR"

echo "========================================================"
echo "Starting Fully-Supervised Training on $NUM_GPUS GPUs..."
echo "Experiment:      $EXP_NAME"
echo "Queue device:    $QUEUE_DEVICE"
echo "Retrieval mode:  $RETRIEVAL_MODE"
echo "Find unused:     ${FIND_UNUSED_PARAMS:-false}"
echo "Progress mode:   $PROGRESS"
echo "========================================================"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_sup.py \
    --seed $SEED \
    --num_workers $NUM_WORKERS \
    --save_dir "$SAVE_DIR" \
    --exp_name "$EXP_NAME" \
    --data_root "$DATA_ROOT" \
    --val_ratio $VAL_RATIO \
    --test_ratio $TEST_RATIO \
    ${SPLIT_CSV:+--split_csv "$SPLIT_CSV"} \
    --img_size 256 256 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --warmup_ratio $WARMUP_RATIO \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --best_window_start $BEST_WINDOW_START \
    --best_window_end $BEST_WINDOW_END \
    --num_classes $NUM_CLASSES \
    --in_chans $IN_CHANS \
    --text_dim $TEXT_DIM \
    --c_spe $C_SPE \
    --c_attn $C_ATTN \
    --router_top_k $ROUTER_TOP_K \
    --stem_ch $STEM_CH \
    --queue_len $QUEUE_LEN \
    --queue_device $QUEUE_DEVICE \
    --retrieval_mode $RETRIEVAL_MODE \
    --queue_chunk_size $QUEUE_CHUNK_SIZE \
    --router_temperature $ROUTER_TEMPERATURE \
    $SMD_EMA_BASES \
    $FIND_UNUSED_PARAMS \
    $DICE_CLASS_WEIGHTS \
    $CE_CLASS_WEIGHTS \
    --ce_weight $CE_WEIGHT \
    --dice_weight $DICE_WEIGHT \
    --lambda_aux_sup $LAMBDA_AUX_SUP \
    --progress $PROGRESS \
    --log_interval $LOG_INTERVAL \
    2>&1 | tee "$SAVE_DIR/train_log.txt"
