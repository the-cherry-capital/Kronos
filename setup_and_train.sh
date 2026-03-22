#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════
# Kronos-mini Pretraining — One-Shot Setup & Launch
# ═══════════════════════════════════════════════════════════════════════
#
# Usage:
#
#   # Single GPU / CPU
#   bash setup_and_train.sh
#
#   # Multi-GPU (single node, 4 GPUs)
#   NUM_GPUS=4 bash setup_and_train.sh
#
#   # Multi-node (run on each node)
#   NUM_GPUS=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash setup_and_train.sh
#   NUM_GPUS=8 NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 bash setup_and_train.sh
#
#   # Skip data download (already have it)
#   SKIP_DOWNLOAD=1 NUM_GPUS=4 bash setup_and_train.sh
#
# ═══════════════════════════════════════════════════════════════════════

REPO_URL="${REPO_URL:-https://github.com/the-cherry-capital/Kronos.git}"
BRANCH="${BRANCH:-pretrain-kronos-mini}"
INSTALL_DIR="${INSTALL_DIR:-$(pwd)/Kronos}"

# Training config
NUM_GPUS="${NUM_GPUS:-0}"          # 0 = single process (CPU or auto-detect 1 GPU)
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

# Training hyperparams (override via env)
TOK_EPOCHS="${TOK_EPOCHS:-3}"
PRED_EPOCHS="${PRED_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_SAMPLES="${MAX_SAMPLES:-1000000}"
NUM_WORKERS="${NUM_WORKERS:-4}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║    Kronos-mini Pretraining — One-Shot Setup                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── 1. Clone & checkout ──────────────────────────────────────────────
if [ ! -d "$INSTALL_DIR/.git" ]; then
    echo "[1/5] Cloning repo..."
    git clone "$REPO_URL" "$INSTALL_DIR"
else
    echo "[1/5] Repo exists, pulling latest..."
    git -C "$INSTALL_DIR" fetch origin
fi
cd "$INSTALL_DIR"
git checkout "$BRANCH"
git pull origin "$BRANCH" || true

# ── 2. Virtual environment ───────────────────────────────────────────
echo "[2/5] Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "  Python: $(python3 --version)"
echo "  Torch:  $(python3 -c 'import torch; print(torch.__version__)')"
echo "  CUDA:   $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# ── 3. Login to wandb ────────────────────────────────────────────────
echo "[3/5] Checking wandb..."
if python3 -c "import wandb; assert wandb.api.api_key" 2>/dev/null; then
    echo "  wandb: logged in"
    WANDB_FLAG=""
else
    echo "  wandb: not logged in — run 'wandb login' to enable logging"
    echo "  Continuing without wandb..."
    WANDB_FLAG="--no_wandb"
fi

# ── 4. Download data ─────────────────────────────────────────────────
if [ "$SKIP_DOWNLOAD" = "0" ]; then
    echo "[4/5] Downloading market data (~200 tickers)..."
    python3 download_data.py
else
    echo "[4/5] Skipping data download (SKIP_DOWNLOAD=1)"
fi

# ── 5. Launch training ───────────────────────────────────────────────
echo "[5/5] Launching pretraining..."

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
    # Distributed launch via torchrun
    TOTAL_PROCS=$((NUM_GPUS > 0 ? NUM_GPUS : 1))
    echo "  Mode: Distributed (${NNODES} nodes × ${TOTAL_PROCS} GPUs)"

    torchrun \
        --nnodes="$NNODES" \
        --nproc_per_node="$TOTAL_PROCS" \
        --node_rank="$NODE_RANK" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        pretrain_scaled.py "${TRAIN_ARGS[@]}"
else
    # Single process
    echo "  Mode: Single process"
    python3 pretrain_scaled.py "${TRAIN_ARGS[@]}"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Done! Checkpoints saved in: $INSTALL_DIR/pretrained_mini_scaled/"
echo "════════════════════════════════════════════════════════════════"
