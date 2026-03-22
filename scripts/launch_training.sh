#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════
# 3/3 — Launch Training
# ═══════════════════════════════════════════════════════════════════════
#
# Launches Kronos-mini pretraining. Supports single process, multi-GPU,
# and multi-node distributed training.
#
# Prerequisite: run scripts/setup_env.sh and scripts/prepare_data.sh first.
#
# Usage:
#   # Single GPU / CPU
#   bash scripts/launch_training.sh
#
#   # Multi-GPU (single node, 4 GPUs)
#   NUM_GPUS=4 bash scripts/launch_training.sh
#
#   # Multi-node (run on each node)
#   NUM_GPUS=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash scripts/launch_training.sh
#   NUM_GPUS=8 NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 bash scripts/launch_training.sh
#
# Env vars:
#   NUM_GPUS      — GPUs per node (0 = single process, default: 0)
#   NNODES        — number of nodes (default: 1)
#   NODE_RANK     — this node's rank (default: 0)
#   MASTER_ADDR   — master node address (default: 127.0.0.1)
#   MASTER_PORT   — master node port (default: 29500)
#   BATCH_SIZE    — per-GPU batch size (default: 64)
#   TOK_EPOCHS    — tokenizer epochs (default: 3)
#   PRED_EPOCHS   — predictor epochs (default: 5)
#   MAX_SAMPLES   — cap training samples (default: 1000000)
#   NUM_WORKERS   — dataloader workers (default: 4)
#   INSTALL_DIR   — repo location (default: ./Kronos)
# ═══════════════════════════════════════════════════════════════════════

INSTALL_DIR="${INSTALL_DIR:-$(pwd)}"

# Auto-detect: if we're inside the repo already, use cwd
if [ ! -f "$INSTALL_DIR/pretrain_scaled.py" ]; then
    INSTALL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi

# Training config
NUM_GPUS="${NUM_GPUS:-0}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Hyperparams
TOK_EPOCHS="${TOK_EPOCHS:-3}"
PRED_EPOCHS="${PRED_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_SAMPLES="${MAX_SAMPLES:-1000000}"
NUM_WORKERS="${NUM_WORKERS:-4}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║    [3/3] Launch Training                                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"

cd "$INSTALL_DIR"
source .venv/bin/activate

# Wandb check
if python3 -c "import wandb; assert wandb.api.api_key" 2>/dev/null; then
    echo "  wandb: logged in"
    WANDB_FLAG=""
else
    echo "  wandb: disabled (run 'wandb login' to enable)"
    WANDB_FLAG="--no_wandb"
fi

TRAIN_ARGS=(
    --tok_epochs "$TOK_EPOCHS"
    --pred_epochs "$PRED_EPOCHS"
    --tok_batch_size "$BATCH_SIZE"
    --pred_batch_size "$BATCH_SIZE"
    --max_samples "$MAX_SAMPLES"
    --num_workers "$NUM_WORKERS"
    $WANDB_FLAG
)

if [ "$NUM_GPUS" -gt 1 ] || [ "$NNODES" -gt 1 ]; then
    TOTAL_PROCS=$((NUM_GPUS > 0 ? NUM_GPUS : 1))
    echo "  Mode:       Distributed (${NNODES} nodes x ${TOTAL_PROCS} GPUs)"
    echo "  Batch:      ${BATCH_SIZE}/GPU x ${TOTAL_PROCS} GPUs x ${NNODES} nodes = $((BATCH_SIZE * TOTAL_PROCS * NNODES)) effective"
    echo "  Tok epochs: $TOK_EPOCHS"
    echo "  Pred epochs: $PRED_EPOCHS"
    echo ""

    torchrun \
        --nnodes="$NNODES" \
        --nproc_per_node="$TOTAL_PROCS" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        pretrain_scaled.py "${TRAIN_ARGS[@]}"
else
    echo "  Mode:       Single process"
    echo "  Batch:      $BATCH_SIZE"
    echo "  Tok epochs: $TOK_EPOCHS"
    echo "  Pred epochs: $PRED_EPOCHS"
    echo ""

    python3 pretrain_scaled.py "${TRAIN_ARGS[@]}"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Done! Checkpoints: $INSTALL_DIR/pretrained_mini_scaled/"
echo "════════════════════════════════════════════════════════════════"
