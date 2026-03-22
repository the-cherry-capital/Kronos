#!/usr/bin/env python3
"""
Kronos-mini distributed pretraining from scratch.

Full PyTorch DDP support for multi-node, multi-GPU training.
Two-stage: Tokenizer (BSQ encoder-decoder) → Predictor (autoregressive).

Launch examples:

  # Single GPU / CPU
  python pretrain_scaled.py

  # Single node, 4 GPUs
  torchrun --standalone --nproc_per_node=4 pretrain_scaled.py

  # Multi-node (2 nodes × 8 GPUs each)
  # -- Node 0 (master) --
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
           --master_addr=10.0.0.1 --master_port=29500 pretrain_scaled.py

  # -- Node 1 --
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
           --master_addr=10.0.0.1 --master_port=29500 pretrain_scaled.py
"""

import os
import sys
import glob
import time
import random
import datetime
import json
import argparse
import socket
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb
except ImportError:
    wandb = None

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from model.kronos import KronosTokenizer, Kronos


# ═══════════════════════════════════════════════════════════════════════
# Distributed Helpers
# ═══════════════════════════════════════════════════════════════════════

def setup_distributed():
    """Initialize DDP from torchrun env vars. Returns (rank, world_size, local_rank, device)."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        # Pick backend: nccl for CUDA, gloo for CPU/MPS
        if torch.cuda.is_available():
            backend = "nccl"
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            backend = "gloo"
            device = torch.device("cpu")

        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    else:
        # Single process — auto-detect best device
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    return rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def all_reduce_scalar(value, device):
    """Sum a scalar across all ranks."""
    if not dist.is_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def print_rank0(msg):
    if is_main_process():
        print(msg)


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

def get_config():
    parser = argparse.ArgumentParser(description="Kronos Distributed Pretraining")

    parser.add_argument("--model_size", type=str, default="large", choices=list(MODEL_CONFIGS.keys()),
                        help="Model size: 'mini' (~4M) or 'large' (~499M)")

    # Paths
    parser.add_argument("--data_dir", type=str, default=os.path.join(ROOT_DIR, "data", "pretrain"))
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # Data
    parser.add_argument("--lookback", type=int, default=128)
    parser.add_argument("--predict", type=int, default=16)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--max_samples", type=int, default=1_000_000)
    parser.add_argument("--num_workers", type=int, default=4)

    # Tokenizer training
    parser.add_argument("--tok_epochs", type=int, default=3)
    parser.add_argument("--tok_batch_size", type=int, default=None, help="Per-GPU batch size")
    parser.add_argument("--tok_lr", type=float, default=None)
    parser.add_argument("--tok_grad_accum", type=int, default=1)

    # Predictor training
    parser.add_argument("--pred_epochs", type=int, default=5)
    parser.add_argument("--pred_batch_size", type=int, default=None, help="Per-GPU batch size")
    parser.add_argument("--pred_lr", type=float, default=None)
    parser.add_argument("--pred_grad_accum", type=int, default=1)

    # General training
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--log_interval", type=int, default=1)

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="kronos")
    parser.add_argument("--wandb_entity", type=str, default="billzhao811")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()
    args.window = args.lookback + args.predict + 1

    # Resolve model-size-dependent defaults
    is_large = args.model_size == "large"
    if args.save_dir is None:
        args.save_dir = os.path.join(ROOT_DIR, f"pretrained_{args.model_size}_scaled")
    if args.tok_batch_size is None:
        args.tok_batch_size = 1024 if is_large else 64
    if args.pred_batch_size is None:
        args.pred_batch_size = 1024 if is_large else 64
    if args.tok_lr is None:
        args.tok_lr = 8e-4 if is_large else 2e-4
    if args.pred_lr is None:
        args.pred_lr = 1.6e-3 if is_large else 4e-4

    args.tok_arch, args.pred_arch = MODEL_CONFIGS[args.model_size]
    return args


# ═══════════════════════════════════════════════════════════════════════
# Model Architecture Constants
# ═══════════════════════════════════════════════════════════════════════

# --- Kronos-mini (4M params, official HuggingFace config) ---

TOK_ARCH_MINI = dict(
    d_in=6, d_model=256, n_heads=4, ff_dim=512,
    n_enc_layers=4, n_dec_layers=4,
    ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
    s1_bits=10, s2_bits=10,
    beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=5,
)

PRED_ARCH_MINI = dict(
    s1_bits=10, s2_bits=10, n_layers=4, d_model=256,
    n_heads=4, ff_dim=512,
    ffn_dropout_p=0.2, attn_dropout_p=0.0, resid_dropout_p=0.2,
    token_dropout_p=0.0, learn_te=True,
)

# --- Kronos-large (~499M params) ---

TOK_ARCH_LARGE = dict(
    d_in=6, d_model=1024, n_heads=16, ff_dim=4120,
    n_enc_layers=8, n_dec_layers=8,
    ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
    s1_bits=10, s2_bits=10,
    beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=5,
)

PRED_ARCH_LARGE = dict(
    s1_bits=10, s2_bits=10, n_layers=15, d_model=1024,
    n_heads=16, ff_dim=4120,
    ffn_dropout_p=0.1, attn_dropout_p=0.0, resid_dropout_p=0.1,
    token_dropout_p=0.0, learn_te=True,
)

MODEL_CONFIGS = {
    "mini": (TOK_ARCH_MINI, PRED_ARCH_MINI),
    "large": (TOK_ARCH_LARGE, PRED_ARCH_LARGE),
}


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

def set_seed(seed, rank=0):
    actual = seed + rank
    random.seed(actual)
    np.random.seed(actual)
    torch.manual_seed(actual)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def fmt(n):
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.1f}K"
    return str(n)


def fmt_time(s):
    return str(datetime.timedelta(seconds=int(s)))


def fmt_bytes(n):
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.2f}{unit}"
        value /= 1024.0


def get_cuda_device_index(device):
    if isinstance(device, torch.device) and device.index is not None:
        return device.index
    return torch.cuda.current_device()


def get_gpu_memory_stats(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return None

    idx = get_cuda_device_index(device)
    allocated = int(torch.cuda.memory_allocated(idx))
    reserved = int(torch.cuda.memory_reserved(idx))
    max_allocated = int(torch.cuda.max_memory_allocated(idx))
    max_reserved = int(torch.cuda.max_memory_reserved(idx))

    try:
        free, total = torch.cuda.mem_get_info(idx)
    except Exception:
        total = int(torch.cuda.get_device_properties(idx).total_memory)
        free = max(total - reserved, 0)

    return {
        "allocated": allocated,
        "reserved": reserved,
        "max_allocated": max_allocated,
        "max_reserved": max_reserved,
        "free": int(free),
        "total": int(total),
    }


def format_gpu_memory(device):
    stats = get_gpu_memory_stats(device)
    if stats is None:
        return "gpu_mem=n/a"

    used = stats["total"] - stats["free"]
    return f"gpu={fmt_bytes(used)}/{fmt_bytes(stats['total'])}"


def gpu_memory_log_data(device, prefix):
    stats = get_gpu_memory_stats(device)
    if stats is None:
        return {}

    gib = 1024 ** 3
    used = stats["total"] - stats["free"]
    return {
        f"{prefix}/gpu_used_gb": used / gib,
        f"{prefix}/gpu_free_gb": stats["free"] / gib,
        f"{prefix}/gpu_total_gb": stats["total"] / gib,
        f"{prefix}/gpu_allocated_gb": stats["allocated"] / gib,
        f"{prefix}/gpu_reserved_gb": stats["reserved"] / gib,
        f"{prefix}/gpu_peak_allocated_gb": stats["max_allocated"] / gib,
        f"{prefix}/gpu_peak_reserved_gb": stats["max_reserved"] / gib,
    }


def reset_gpu_peak_memory(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return

    idx = get_cuda_device_index(device)
    try:
        torch.cuda.reset_peak_memory_stats(idx)
    except Exception:
        torch.cuda.reset_peak_memory_stats()


# ═══════════════════════════════════════════════════════════════════════
# Multi-Stock Dataset
# ═══════════════════════════════════════════════════════════════════════

class MultiStockKlineDataset(Dataset):
    """
    Multi-stock sliding-window dataset for pretraining.
    Windows never cross stock boundaries. Instance-normalized per window.
    """

    FEAT = ['open', 'high', 'low', 'close', 'volume', 'amount']
    TIME = ['minute', 'hour', 'weekday', 'day', 'month']

    def __init__(self, data_dir, split, window, clip, train_ratio, max_samples, seed):
        self.window = window
        self.clip = clip
        self._epoch = 0

        csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

        self.stocks_feat = []
        self.stocks_time = []
        self.flat_index = []
        total_rows = 0

        for csv_path in csv_files:
            if not csv_path.endswith(".csv"):
                continue
            df = pd.read_csv(csv_path)
            if 'timestamps' not in df.columns:
                continue

            df['timestamps'] = pd.to_datetime(df['timestamps'])
            df = df.sort_values('timestamps').reset_index(drop=True)
            df['minute'] = df['timestamps'].dt.minute
            df['hour'] = df['timestamps'].dt.hour
            df['weekday'] = df['timestamps'].dt.weekday
            df['day'] = df['timestamps'].dt.day
            df['month'] = df['timestamps'].dt.month

            if 'amount' not in df.columns:
                df['amount'] = df.get('volume', 0) * df[['open', 'high', 'low', 'close']].mean(axis=1)
            df = df.ffill().fillna(0)

            n = len(df)
            cut = int(n * train_ratio)
            df = df.iloc[:cut] if split == 'train' else df.iloc[cut:]

            if len(df) < window:
                continue

            sid = len(self.stocks_feat)
            self.stocks_feat.append(df[self.FEAT].values.astype(np.float32))
            self.stocks_time.append(df[self.TIME].values.astype(np.float32))

            for j in range(len(df) - window + 1):
                self.flat_index.append((sid, j))
            total_rows += len(df)

        if max_samples and len(self.flat_index) > max_samples:
            self.flat_index = random.Random(seed).sample(self.flat_index, max_samples)

        if is_main_process():
            print(f"  [{split.upper()}] stocks={len(self.stocks_feat)}, "
                  f"rows={total_rows:,}, samples={len(self.flat_index):,}")

    def set_epoch_seed(self, epoch):
        self._epoch = epoch

    def __len__(self):
        return len(self.flat_index)

    def __getitem__(self, idx):
        sid, offset = self.flat_index[idx]
        feat = self.stocks_feat[sid]
        max_start = len(feat) - self.window
        start = (offset + self._epoch * 7919) % (max_start + 1)
        end = start + self.window

        x = feat[start:end].copy()
        stamp = self.stocks_time[sid][start:end].copy()

        mean, std = x.mean(0), x.std(0)
        x = np.clip((x - mean) / (std + 1e-5), -self.clip, self.clip)

        return torch.from_numpy(x), torch.from_numpy(stamp)


# ═══════════════════════════════════════════════════════════════════════
# Dataloader Factory
# ═══════════════════════════════════════════════════════════════════════

def make_loaders(args, rank, world_size):
    """Create train/val datasets and distributed dataloaders."""
    train_ds = MultiStockKlineDataset(
        args.data_dir, 'train', args.window, args.clip,
        args.train_ratio, args.max_samples, args.seed)
    val_ds = MultiStockKlineDataset(
        args.data_dir, 'val', args.window, args.clip,
        args.train_ratio, max_samples=50_000, seed=args.seed + 1)

    use_ddp = dist.is_initialized()

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    ) if use_ddp else None

    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False
    ) if use_ddp else None

    train_loader = DataLoader(
        train_ds, batch_size=args.tok_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.tok_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader, train_ds, val_ds, train_sampler, val_sampler


# ═══════════════════════════════════════════════════════════════════════
# Stage 1: Tokenizer Pretraining
# ═══════════════════════════════════════════════════════════════════════

def train_tokenizer(args, device, rank, local_rank, world_size, wandb_run):
    print_rank0("\n" + "=" * 64)
    print_rank0("  STAGE 1: Tokenizer Pretraining")
    print_rank0("=" * 64)

    tokenizer = KronosTokenizer(**args.tok_arch).to(device)
    print_rank0(f"  Tokenizer params: {fmt(count_params(tokenizer))}")
    print_rank0(f"  GPU memory @ stage start: {format_gpu_memory(device)}")

    # Wrap in DDP
    if dist.is_initialized():
        tokenizer = DDP(tokenizer, device_ids=[local_rank] if device.type == "cuda" else None,
                        find_unused_parameters=False)

    raw_model = tokenizer.module if dist.is_initialized() else tokenizer

    # Data
    print_rank0("  Loading data...")
    train_loader, val_loader, train_ds, _, train_sampler, _ = make_loaders(args, rank, world_size)

    effective_bs = args.tok_batch_size * world_size * args.tok_grad_accum
    steps_per_epoch = len(train_loader) // args.tok_grad_accum
    total_steps = steps_per_epoch * args.tok_epochs
    print_rank0(f"  Per-GPU batch: {args.tok_batch_size}, World: {world_size}, "
                f"Grad accum: {args.tok_grad_accum}, Effective batch: {effective_bs}")
    print_rank0(f"  Batches/epoch: {len(train_loader):,}  Steps/epoch: {steps_per_epoch:,}  Total steps: {total_steps:,}")

    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=args.tok_lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.tok_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=args.tok_epochs, pct_start=0.05, div_factor=10,
    )

    best_val_loss = float('inf')
    save_path = os.path.join(args.save_dir, "tokenizer", "best_model")
    if is_main_process():
        os.makedirs(save_path, exist_ok=True)
    barrier()

    global_step = 0
    t0 = time.time()

    for epoch in range(args.tok_epochs):
        epoch_t0 = time.time()
        tokenizer.train()
        reset_gpu_peak_memory(device)
        train_ds.set_epoch_seed(epoch * 10000 + rank)
        if train_sampler:
            train_sampler.set_epoch(epoch)

        epoch_loss, epoch_recon, epoch_bsq, epoch_steps = 0.0, 0.0, 0.0, 0
        optimizer.zero_grad()

        for batch_idx, (batch_x, _) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)

            (z_pre, z), bsq_loss, _, _ = (raw_model if not dist.is_initialized() else tokenizer)(batch_x)

            recon_pre = F.mse_loss(z_pre, batch_x)
            recon_all = F.mse_loss(z, batch_x)
            recon = recon_pre + recon_all
            loss = (recon + bsq_loss) / 2

            (loss / args.tok_grad_accum).backward()

            if (batch_idx + 1) % args.tok_grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), max_norm=2.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main_process() and wandb_run and global_step % args.log_interval == 0:
                    log_data = {
                        "tokenizer/train_loss": loss.item(),
                        "tokenizer/recon_loss": recon.item(),
                        "tokenizer/bsq_loss": bsq_loss.item(),
                        "tokenizer/lr": optimizer.param_groups[0]['lr'],
                    }
                    log_data.update(gpu_memory_log_data(device, "tokenizer"))
                    wandb_run.log(log_data, step=global_step)

                if is_main_process() and global_step % args.log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    pct = global_step / total_steps * 100
                    print(f"    [Epoch {epoch+1}/{args.tok_epochs}] "
                          f"[Step {global_step:,}/{total_steps:,} ({pct:.1f}%)] "
                          f"loss={loss.item():.4f} | recon={recon.item():.4f} | "
                          f"bsq={bsq_loss.item():.4f} | lr={lr:.6f} | "
                          f"elapsed={fmt_time(time.time() - t0)} | "
                          f"{format_gpu_memory(device)}")

            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_bsq += bsq_loss.item()
            epoch_steps += 1

        # ── Validation (all ranks compute, then reduce) ──────────────
        tokenizer.eval()
        val_loss_sum, val_count = 0.0, 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                (_, z), _, _, _ = (raw_model if not dist.is_initialized() else tokenizer)(batch_x)
                val_loss_sum += F.mse_loss(z, batch_x).item() * batch_x.size(0)
                val_count += batch_x.size(0)

        val_loss_sum = all_reduce_scalar(val_loss_sum, device)
        val_count = all_reduce_scalar(val_count, device)
        avg_val = val_loss_sum / val_count if val_count > 0 else 0

        if is_main_process():
            avg_train = epoch_loss / epoch_steps
            pct = global_step / total_steps * 100
            print(f"  ── Epoch {epoch+1}/{args.tok_epochs} done ── "
                  f"step={global_step:,}/{total_steps:,} ({pct:.1f}%) | "
                  f"train={avg_train:.4f} | val={avg_val:.4f} | "
                  f"time={fmt_time(time.time() - epoch_t0)} | "
                  f"{format_gpu_memory(device)}")

            if wandb_run:
                log_data = {
                    "tokenizer/epoch": epoch + 1,
                    "tokenizer/epoch_train_loss": avg_train,
                    "tokenizer/epoch_val_loss": avg_val,
                    "tokenizer/best_val_loss": min(best_val_loss, avg_val),
                }
                log_data.update(gpu_memory_log_data(device, "tokenizer"))
                wandb_run.log(log_data, step=global_step)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                raw_model.save_pretrained(save_path)
                print(f"  ** Saved best tokenizer (val={best_val_loss:.4f})")

        barrier()

    if is_main_process():
        print(f"\n  Stage 1 done. Best val={best_val_loss:.4f}  Time: {fmt_time(time.time() - t0)}")

    return save_path, global_step


# ═══════════════════════════════════════════════════════════════════════
# Stage 2: Predictor Pretraining
# ═══════════════════════════════════════════════════════════════════════

def train_predictor(args, device, rank, local_rank, world_size, wandb_run, tokenizer_path, step_offset):
    print_rank0("\n" + "=" * 64)
    print_rank0("  STAGE 2: Predictor Pretraining (Kronos-mini)")
    print_rank0("=" * 64)
    print_rank0(f"  GPU memory @ stage start: {format_gpu_memory(device)}")

    # Frozen tokenizer (no DDP needed — inference only)
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer.eval().to(device)
    for p in tokenizer.parameters():
        p.requires_grad = False
    print_rank0(f"  Loaded frozen tokenizer from {tokenizer_path}")

    # Predictor
    model = Kronos(**args.pred_arch).to(device)
    print_rank0(f"  Predictor params: {fmt(count_params(model))}")

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                    find_unused_parameters=False)

    raw_model = model.module if dist.is_initialized() else model

    # Data
    print_rank0("  Loading data...")

    # Re-create loaders with predictor batch size
    args_copy = argparse.Namespace(**vars(args))
    args_copy.tok_batch_size = args.pred_batch_size  # reuse loader factory
    train_loader, val_loader, train_ds, _, train_sampler, _ = make_loaders(args_copy, rank, world_size)

    effective_bs = args.pred_batch_size * world_size * args.pred_grad_accum
    steps_per_epoch = len(train_loader) // args.pred_grad_accum
    total_steps = steps_per_epoch * args.pred_epochs
    print_rank0(f"  Per-GPU batch: {args.pred_batch_size}, World: {world_size}, "
                f"Grad accum: {args.pred_grad_accum}, Effective batch: {effective_bs}")
    print_rank0(f"  Batches/epoch: {len(train_loader):,}  Steps/epoch: {steps_per_epoch:,}  Total steps: {total_steps:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.pred_lr,
        betas=(0.9, 0.95), weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.pred_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=args.pred_epochs, pct_start=0.05, div_factor=10,
    )

    best_val_loss = float('inf')
    save_path = os.path.join(args.save_dir, "predictor", "best_model")
    if is_main_process():
        os.makedirs(save_path, exist_ok=True)
    barrier()

    global_step = 0
    t0 = time.time()

    for epoch in range(args.pred_epochs):
        epoch_t0 = time.time()
        model.train()
        reset_gpu_peak_memory(device)
        train_ds.set_epoch_seed(epoch * 10000 + rank)
        if train_sampler:
            train_sampler.set_epoch(epoch)

        ep_loss, ep_s1, ep_s2, ep_steps = 0.0, 0.0, 0.0, 0
        optimizer.zero_grad()

        for batch_idx, (batch_x, batch_stamp) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_stamp = batch_stamp.to(device, non_blocking=True)

            with torch.no_grad():
                tok_s1, tok_s2 = tokenizer.encode(batch_x, half=True)

            s1_in, s2_in = tok_s1[:, :-1], tok_s2[:, :-1]
            s1_tgt, s2_tgt = tok_s1[:, 1:], tok_s2[:, 1:]
            stamp_in = batch_stamp[:, :-1, :]

            s1_logits, s2_logits = (raw_model if not dist.is_initialized() else model)(
                s1_in, s2_in, stamp_in,
                use_teacher_forcing=True, s1_targets=s1_tgt,
            )
            loss, s1_loss, s2_loss = raw_model.head.compute_loss(
                s1_logits, s2_logits, s1_tgt, s2_tgt
            )

            (loss / args.pred_grad_accum).backward()

            if (batch_idx + 1) % args.pred_grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                abs_step = step_offset + global_step
                if is_main_process() and wandb_run and global_step % args.log_interval == 0:
                    log_data = {
                        "predictor/train_loss": loss.item(),
                        "predictor/s1_loss": s1_loss.item(),
                        "predictor/s2_loss": s2_loss.item(),
                        "predictor/lr": optimizer.param_groups[0]['lr'],
                    }
                    log_data.update(gpu_memory_log_data(device, "predictor"))
                    wandb_run.log(log_data, step=abs_step)

                if is_main_process() and global_step % args.log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    pct = global_step / total_steps * 100
                    print(f"    [Epoch {epoch+1}/{args.pred_epochs}] "
                          f"[Step {global_step:,}/{total_steps:,} ({pct:.1f}%)] "
                          f"loss={loss.item():.4f} | s1={s1_loss.item():.4f} | "
                          f"s2={s2_loss.item():.4f} | lr={lr:.6f} | "
                          f"elapsed={fmt_time(time.time() - t0)} | "
                          f"{format_gpu_memory(device)}")

            ep_loss += loss.item()
            ep_s1 += s1_loss.item()
            ep_s2 += s2_loss.item()
            ep_steps += 1

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        vl_sum, vs1_sum, vs2_sum, v_count = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch_x, batch_stamp in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_stamp = batch_stamp.to(device, non_blocking=True)

                tok_s1, tok_s2 = tokenizer.encode(batch_x, half=True)
                s1_in, s2_in = tok_s1[:, :-1], tok_s2[:, :-1]
                s1_tgt, s2_tgt = tok_s1[:, 1:], tok_s2[:, 1:]
                stamp_in = batch_stamp[:, :-1, :]

                s1_logits, s2_logits = (raw_model if not dist.is_initialized() else model)(
                    s1_in, s2_in, stamp_in,
                )
                v_loss, v_s1, v_s2 = raw_model.head.compute_loss(
                    s1_logits, s2_logits, s1_tgt, s2_tgt
                )
                bs = batch_x.size(0)
                vl_sum += v_loss.item() * bs
                vs1_sum += v_s1.item() * bs
                vs2_sum += v_s2.item() * bs
                v_count += bs

        vl_sum = all_reduce_scalar(vl_sum, device)
        vs1_sum = all_reduce_scalar(vs1_sum, device)
        vs2_sum = all_reduce_scalar(vs2_sum, device)
        v_count = all_reduce_scalar(v_count, device)
        avg_val = vl_sum / v_count if v_count > 0 else 0

        if is_main_process():
            avg_train = ep_loss / ep_steps
            pct = global_step / total_steps * 100
            print(f"  ── Epoch {epoch+1}/{args.pred_epochs} done ── "
                  f"step={global_step:,}/{total_steps:,} ({pct:.1f}%) | "
                  f"train={avg_train:.4f} (s1={ep_s1/ep_steps:.4f}, s2={ep_s2/ep_steps:.4f}) | "
                  f"val={avg_val:.4f} | time={fmt_time(time.time() - epoch_t0)} | "
                  f"{format_gpu_memory(device)}")

            if wandb_run:
                abs_step = step_offset + global_step
                log_data = {
                    "predictor/epoch": epoch + 1,
                    "predictor/epoch_train_loss": avg_train,
                    "predictor/epoch_val_loss": avg_val,
                    "predictor/best_val_loss": min(best_val_loss, avg_val),
                }
                log_data.update(gpu_memory_log_data(device, "predictor"))
                wandb_run.log(log_data, step=abs_step)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                raw_model.save_pretrained(save_path)
                print(f"  ** Saved best predictor (val={best_val_loss:.4f})")

        barrier()

    if is_main_process():
        print(f"\n  Stage 2 done. Best val={best_val_loss:.4f}  Time: {fmt_time(time.time() - t0)}")

    return save_path


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = get_config()
    rank, world_size, local_rank, device = setup_distributed()
    set_seed(args.seed, rank)

    print_rank0("╔══════════════════════════════════════════════════════════════╗")
    print_rank0(f"║    Kronos-{args.model_size} Distributed Pretraining" + " " * max(0, 38 - len(args.model_size)) + "║")
    print_rank0("╚══════════════════════════════════════════════════════════════╝")
    print_rank0(f"  Model:      {args.model_size}")
    print_rank0(f"  Rank: {rank}/{world_size}  Device: {device}")
    print_rank0(f"  Host:       {socket.gethostname()}  Local rank: {local_rank}")
    print_rank0(f"  Python:     {sys.version.split()[0]}  Torch: {torch.__version__}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(get_cuda_device_index(device))
        print_rank0(f"  GPU:        {props.name}  VRAM: {fmt_bytes(props.total_memory)}  CUDA: {torch.version.cuda}")
        print_rank0(f"  GPU memory: {format_gpu_memory(device)}")
    print_rank0(f"  Data:       {args.data_dir}")
    print_rank0(f"  Save to:    {args.save_dir}")
    print_rank0(f"  Max samples: {args.max_samples:,}")

    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)
    barrier()

    # Wandb (rank 0 only)
    wandb_run = None
    if is_main_process() and not args.no_wandb and wandb is not None:
        run_name = args.wandb_run_name or f"kronos-{args.model_size}-pretrain-{world_size}gpu"
        config_dict = {
            "model_size": args.model_size,
            "tokenizer_arch": args.tok_arch,
            "predictor_arch": args.pred_arch,
            "training": vars(args),
            "world_size": world_size,
        }
        wandb_run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=run_name, config=config_dict,
            tags=["pretrain", "kronos-mini", "ddp", f"{world_size}gpu"],
        )

        with open(os.path.join(args.save_dir, "pretrain_config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

    # ── Stage 1 ───────────────────────────────────────────────────────
    tok_path, tok_steps = train_tokenizer(args, device, rank, local_rank, world_size, wandb_run)

    # ── Stage 2 ───────────────────────────────────────────────────────
    pred_path = train_predictor(args, device, rank, local_rank, world_size, wandb_run, tok_path, tok_steps)

    # ── Done ──────────────────────────────────────────────────────────
    if wandb_run:
        wandb_run.finish()

    print_rank0("\n" + "=" * 64)
    print_rank0("  PRETRAINING COMPLETE")
    print_rank0("=" * 64)
    print_rank0(f"  Tokenizer: {tok_path}")
    print_rank0(f"  Predictor: {pred_path}")
    print_rank0("=" * 64)

    cleanup_distributed()


if __name__ == "__main__":
    main()
