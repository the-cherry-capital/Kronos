#!/bin/bash
#SBATCH -A llmservice_fm_text
#SBATCH --qos=normal
#SBATCH -p batch_long
#SBATCH -N 4
#SBATCH -t 2:00:00
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --job-name=bash
#SBATCH --output=/home/tolong/Kronos/slurm_logs/%x_%j.out
#SBATCH --error=/home/tolong/Kronos/slurm_logs/%x_%j.err

set -euo pipefail

# Edit these few lines and submit with:
#   sbatch scripts/submit_large_scale_training.sh

export INSTALL_DIR="/home/tolong/Kronos"
cd "$INSTALL_DIR"

export NUM_GPUS=4
export NNODES="$SLURM_NNODES"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | awk 'NR == 1 { print; exit }')"
export MASTER_PORT=29500

export TOK_EPOCHS=10
export PRED_EPOCHS=10
export MAX_SAMPLES=1000000
export NUM_WORKERS=4

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
echo "gpus_per_node=$NUM_GPUS batch_size=${BATCH_SIZE:-auto} max_samples=$MAX_SAMPLES"
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
