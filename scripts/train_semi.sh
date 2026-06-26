#!/bin/bash

# ============================================================
# LoTS-Net 半监督训练脚本
# ============================================================

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
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
NUM_WORKERS=8

# ---- 数据 ----
DATA_ROOT="data/MDC_Resized_256_320_to_256_256_overlap_0_0"          # 请替换为实际数据路径
LABEL_RATIO=0.2

# ---- 数据集划分模式 ----
# ""                           → 随机划分（默认），按比例随机分四路
# "splits/split_semi.csv"      → CSV 预设划分（推荐），由 dataview.ipynb 生成
#                                CSV 需含 'case_id' 和 'split'
#                                （train_labeled/train_unlabeled/val/test）列
#                                设置后，LABEL_RATIO/val/test 比例被忽略
SPLIT_CSV=""
PL_CONF_THRESH=0.70
BATCH_SIZE=1

# ---- 训练超参数 ----
EPOCHS=100
LR=0.0003
WARMUP_RATIO=0.4
PLATEAU_RATIO=0.1
RAMPUP_RATIO=0.333
LAMBDA_AUX=0.3
GAMMA=0.5
LAMBDA_MAX=0.4

# ---- 模型超参数 ----
IN_CHANS=60
C_SPE=144
C_ATTN=144
STEM_CH=128
QUEUE_LEN=750
QUEUE_DEVICE="cuda"         # "cpu" 或 "cuda"（cuda 可减少 CPU→GPU 传输延迟）
ROUTER_TOP_K=15
TEXT_ENCODER="bert-large-cased"
TEXT_MODE="default"
ROUTER_SCORE_NOISE_STD=0.05
SMD_EMA_BASES="--smd_ema_bases"       # 设为 "" 可关闭

# ---- 队列检索模式 ----
# "top1"    : 论文 Eq.1，全局池化后 argmax 选最匹配条目，显存极低（推荐）
# "chunked" : Online-Softmax 分块密集注意力，与原版数值等价但显存可控
RETRIEVAL_MODE="top1"
QUEUE_CHUNK_SIZE=32         # 仅 chunked 模式有效：每块处理的队列条目数

# ---- DDP 选项 ----
# 设为 "--find_unused_parameters" 开启（适合消融实验有未使用参数时）
# 设为 ""                        关闭（全模型训练时关闭可降低通信开销）
FIND_UNUSED_PARAMS=""

# ---- Dice Loss 类别权重 ----
# ""                                  → 不传参数，背景/前景等权（默认行为）
# "--dice_class_weights 0.0 1.0"      → 纯前景 Dice（完全忽略背景）
# "--dice_class_weights 0.3 1.0"      → 前景权重约为背景的 3.3 倍
# "--dice_class_weights 0.5 1.0"      → 前景权重为背景的 2 倍（推荐起点）
DICE_CLASS_WEIGHTS=""

# ---- CE Loss 类别权重 ----
# ""                                  → 不传参数，使用默认 [1.0, 1.0]（等权）
# "--ce_class_weights 1.0 4.0"        → 前景权重为背景的 4 倍
# "--ce_class_weights 1.0 6.0"        → 逆频率加权（适合前景约占 15% 时）
# 建议先运行 dataview.ipynb 得到实际前景占比 p，再设置 (1-p)/p
CE_CLASS_WEIGHTS=""

# ---- CE/Dice 整体系数 ----
# loss = ce_weight*CE + dice_weight*Dice
CE_WEIGHT=1.0
DICE_WEIGHT=1.0

# ---- 消融实验控制 ----
MODEL_VER="v4"
USE_L_PL="--use_l_pl"
USE_L_CON="--use_l_con"

# ---- 日志 ----
PROGRESS="tqdm"              # tqdm / log / none
LOG_INTERVAL=20
NUM_TOP_K_BATCHES=20
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")
EXP_NAME="Semi_${MODEL_VER}_PL_CON_${LABEL_RATIO}_${TIMESTAMP}"
SAVE_DIR="./records/${EXP_NAME}"
mkdir -p "$SAVE_DIR"

echo "========================================================"
echo "Starting Semi-Supervised Training on $NUM_GPUS GPUs..."
echo "Experiment:      $EXP_NAME"
echo "Queue device:    $QUEUE_DEVICE"
echo "Retrieval mode:  $RETRIEVAL_MODE"
echo "Find unused:     ${FIND_UNUSED_PARAMS:-false}"
echo "Progress mode:   $PROGRESS"
echo "========================================================"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_semi.py \
    --seed $SEED \
    --num_workers $NUM_WORKERS \
    --save_dir "$SAVE_DIR" \
    --exp_name "$EXP_NAME" \
    --data_root "$DATA_ROOT" \
    --label_ratio $LABEL_RATIO \
    ${SPLIT_CSV:+--split_csv "$SPLIT_CSV"} \
    --model_version $MODEL_VER \
    $USE_L_PL \
    $USE_L_CON \
    --img_size 256 256 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --in_chans $IN_CHANS \
    --c_spe $C_SPE \
    --c_attn $C_ATTN \
    --stem_ch $STEM_CH \
    --queue_len $QUEUE_LEN \
    --queue_device $QUEUE_DEVICE \
    --retrieval_mode $RETRIEVAL_MODE \
    --queue_chunk_size $QUEUE_CHUNK_SIZE \
    $FIND_UNUSED_PARAMS \
    $DICE_CLASS_WEIGHTS \
    $CE_CLASS_WEIGHTS \
    --ce_weight $CE_WEIGHT \
    --dice_weight $DICE_WEIGHT \
    --router_top_k $ROUTER_TOP_K \
    --pl_conf_thresh $PL_CONF_THRESH \
    --warmup_ratio $WARMUP_RATIO \
    --plateau_ratio $PLATEAU_RATIO \
    --rampup_ratio $RAMPUP_RATIO \
    --lambda_aux $LAMBDA_AUX \
    --lambda_max $LAMBDA_MAX \
    --gamma $GAMMA \
    --router_score_noise_std $ROUTER_SCORE_NOISE_STD \
    --num_top_k_batches $NUM_TOP_K_BATCHES \
    $SMD_EMA_BASES \
    --text_encoder $TEXT_ENCODER \
    --text_mode $TEXT_MODE \
    --progress $PROGRESS \
    --log_interval $LOG_INTERVAL \
    2>&1 | tee "${SAVE_DIR}/train_log.txt"
