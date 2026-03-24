#!/bin/bash
#SBATCH -A llmservice_fm_text
#SBATCH --qos=normal
#SBATCH -p batch_long
#SBATCH -N 4
# Default node count; override with `sbatch -N <nodes>` or `bash scripts/submit_large_scale_training.sh <nodes>`.
#SBATCH -t 8:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --job-name=bash
#SBATCH --output=/home/tolong/Kronos/slurm_logs/%x_%j.out
#SBATCH --error=/home/tolong/Kronos/slurm_logs/%x_%j.err

set -euo pipefail

# Submit with either:
#   sbatch -N <nodes> scripts/submit_large_scale_training.sh
# or:
#   bash scripts/submit_large_scale_training.sh <nodes>

if [ -z "${SLURM_JOB_ID:-}" ]; then
    requested_nodes="${1:-}"
    if [ -z "$requested_nodes" ]; then
        echo "Usage:"
        echo "  sbatch -N <nodes> scripts/submit_large_scale_training.sh"
        echo "  bash scripts/submit_large_scale_training.sh <nodes>"
        exit 1
    fi
    if ! [[ "$requested_nodes" =~ ^[1-9][0-9]*$ ]]; then
        echo "Invalid node count: '$requested_nodes'"
        exit 1
    fi
    exec sbatch --export=ALL -N "$requested_nodes" "$0"
fi

export INSTALL_DIR="/home/tolong/Kronos"
cd "$INSTALL_DIR"

export NUM_GPUS=4
export NNODES="$SLURM_NNODES"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | awk 'NR == 1 { print; exit }')"
export MASTER_PORT=29500

export TOK_EPOCHS="${TOK_EPOCHS:-10}"
export PRED_EPOCHS="${PRED_EPOCHS:-10}"
export BATCH_SIZE="${BATCH_SIZE:-}"
export SEQ_LEN="${SEQ_LEN:-144}"
export TRAIN_RATIO="${TRAIN_RATIO:-0.7}"
export MAX_SAMPLES="${MAX_SAMPLES:-1000000}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export DATA_DIR="${DATA_DIR:-$INSTALL_DIR/data/pretrain}"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NCCL_DEBUG=WARN

echo "===== Slurm job ====="
echo "job_id=$SLURM_JOB_ID"
echo "job_name=$SLURM_JOB_NAME"
echo "nodes=$SLURM_JOB_NODELIST"
echo "nnodes=$SLURM_NNODES"
echo "master=$MASTER_ADDR:$MASTER_PORT"
echo "gpus_per_node=$NUM_GPUS batch_size=${BATCH_SIZE:-auto} seq_len=$SEQ_LEN train_ratio=$TRAIN_RATIO max_samples=$MAX_SAMPLES"
echo "data_dir=$DATA_DIR"
echo "tok_epochs=$TOK_EPOCHS pred_epochs=$PRED_EPOCHS num_workers=$NUM_WORKERS"
scontrol show hostnames "$SLURM_JOB_NODELIST"
echo "====================="

srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 --label \
    bash -lc '
        export NODE_RANK="$SLURM_NODEID"
        echo "===== node $NODE_RANK $(hostname) ====="
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
        fi
        if command -v free >/dev/null 2>&1; then
            free -h
        fi
        bash "$INSTALL_DIR/scripts/launch_training.sh"
    '
