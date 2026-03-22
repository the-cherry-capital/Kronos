#!/usr/bin/env python3
"""
Kronos-mini pretraining from scratch (CPU-only, single-process).

Two-stage pretraining following the Kronos framework:
  Stage 1 — Train KronosTokenizer: learns to encode/decode K-line (OHLCV) data
             into hierarchical discrete tokens via Binary Spherical Quantization.
  Stage 2 — Train Kronos predictor: autoregressive transformer that predicts
             the next token in the quantized sequence.

Dataset: examples/data/XSHG_5min_600977.csv (Shanghai 5-min K-lines, ~2500 bars)

Usage:
    source .venv/bin/activate
    python pretrain_mini.py
"""

import os
import sys
import time
import random
import datetime
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb

# Ensure project root is importable
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from model.kronos import KronosTokenizer, Kronos


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

SEED = 42
DATA_PATH = os.path.join(ROOT_DIR, "examples/data/XSHG_5min_600977.csv")
SAVE_DIR = os.path.join(ROOT_DIR, "pretrained_mini")
DEVICE = "cpu"

# ── Data ──────────────────────────────────────────────────────────────
LOOKBACK_WINDOW = 128
PREDICT_WINDOW = 16
CLIP = 5.0
TRAIN_RATIO = 0.85
VAL_RATIO = 0.15

# ── Tokenizer architecture (Kronos-Tokenizer-2k official config) ─────
TOK_D_IN = 6           # OHLCV + amount
TOK_D_MODEL = 256
TOK_N_HEADS = 4
TOK_FF_DIM = 512
TOK_N_ENC_LAYERS = 4
TOK_N_DEC_LAYERS = 4
TOK_S1_BITS = 10
TOK_S2_BITS = 10
TOK_BSQ_BETA = 0.05
TOK_BSQ_GAMMA0 = 1.0
TOK_BSQ_GAMMA = 1.1
TOK_BSQ_ZETA = 0.05
TOK_BSQ_GROUP_SIZE = 5   # codebook_dim=20, 20%5==0

# ── Predictor architecture (Kronos-mini official config, 4.1M params) ─
PRED_S1_BITS = 10
PRED_S2_BITS = 10
PRED_N_LAYERS = 4
PRED_D_MODEL = 256
PRED_N_HEADS = 4       # head_dim = 64
PRED_FF_DIM = 512
PRED_FFN_DROPOUT = 0.2
PRED_ATTN_DROPOUT = 0.0
PRED_RESID_DROPOUT = 0.2
PRED_TOKEN_DROPOUT = 0.0
PRED_LEARN_TE = True

# ── Training — Stage 1 (Tokenizer) ───────────────────────────────────
TOK_EPOCHS = 15
TOK_BATCH_SIZE = 16
TOK_LR = 2e-4
TOK_WEIGHT_DECAY = 0.1
TOK_LOG_INTERVAL = 20

# ── Training — Stage 2 (Predictor) ───────────────────────────────────
PRED_EPOCHS = 20
PRED_BATCH_SIZE = 16
PRED_LR = 4e-4
PRED_BETA1 = 0.9
PRED_BETA2 = 0.95
PRED_WEIGHT_DECAY = 0.1
PRED_LOG_INTERVAL = 20

# ── Wandb ─────────────────────────────────────────────────────────────
WANDB_PROJECT = "kronos"
WANDB_ENTITY = "billzhao811"


# ═══════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def fmt_params(n):
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def fmt_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════

class KlineDataset(Dataset):
    """Sliding-window dataset over K-line CSV data."""

    FEATURE_COLS = ['open', 'high', 'low', 'close', 'volume', 'amount']
    TIME_COLS = ['minute', 'hour', 'weekday', 'day', 'month']

    def __init__(self, data_path, split='train', lookback=128, predict=16,
                 clip=5.0, train_ratio=0.85, seed=42):
        df = pd.read_csv(data_path)
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        df = df.sort_values('timestamps').reset_index(drop=True)

        # Derive time features
        df['minute'] = df['timestamps'].dt.minute
        df['hour'] = df['timestamps'].dt.hour
        df['weekday'] = df['timestamps'].dt.weekday
        df['day'] = df['timestamps'].dt.day
        df['month'] = df['timestamps'].dt.month

        # Ensure volume/amount exist
        if 'volume' not in df.columns:
            df['volume'] = 0.0
        if 'amount' not in df.columns:
            df['amount'] = df['volume'] * df[['open','high','low','close']].mean(axis=1)

        df = df.fillna(method='ffill').fillna(0)

        # Chronological split
        n = len(df)
        train_end = int(n * train_ratio)

        if split == 'train':
            df = df.iloc[:train_end].reset_index(drop=True)
        else:
            df = df.iloc[train_end:].reset_index(drop=True)

        self.features = df[self.FEATURE_COLS].values.astype(np.float32)
        self.time_feats = df[self.TIME_COLS].values.astype(np.float32)
        self.window = lookback + predict + 1
        self.clip = clip
        self.n_samples = max(0, len(df) - self.window + 1)
        self.seed = seed
        self._epoch = 0

        print(f"  [{split.upper()}] rows={len(df)}, samples={self.n_samples}, window={self.window}")

    def set_epoch_seed(self, epoch):
        self._epoch = epoch

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Deterministic shuffling for training diversity
        max_start = len(self.features) - self.window
        start = (idx * 9973 + (self._epoch + 1) * 104729) % (max_start + 1)
        end = start + self.window

        x = self.features[start:end].copy()
        stamp = self.time_feats[start:end].copy()

        # Instance normalization
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        x = (x - mean) / (std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        return torch.from_numpy(x), torch.from_numpy(stamp)


# ═══════════════════════════════════════════════════════════════════════
# Stage 1: Tokenizer Pretraining
# ═══════════════════════════════════════════════════════════════════════

def train_tokenizer():
    print("\n" + "=" * 64)
    print("  STAGE 1: Tokenizer Pretraining")
    print("=" * 64)

    tokenizer = KronosTokenizer(
        d_in=TOK_D_IN,
        d_model=TOK_D_MODEL,
        n_heads=TOK_N_HEADS,
        ff_dim=TOK_FF_DIM,
        n_enc_layers=TOK_N_ENC_LAYERS,
        n_dec_layers=TOK_N_DEC_LAYERS,
        ffn_dropout_p=0.0,
        attn_dropout_p=0.0,
        resid_dropout_p=0.0,
        s1_bits=TOK_S1_BITS,
        s2_bits=TOK_S2_BITS,
        beta=TOK_BSQ_BETA,
        gamma0=TOK_BSQ_GAMMA0,
        gamma=TOK_BSQ_GAMMA,
        zeta=TOK_BSQ_ZETA,
        group_size=TOK_BSQ_GROUP_SIZE,
    ).to(DEVICE)

    total, trainable = count_params(tokenizer)
    print(f"  Tokenizer params: {fmt_params(total)} (trainable: {fmt_params(trainable)})")

    # Data
    print("  Loading data...")
    train_ds = KlineDataset(DATA_PATH, 'train', LOOKBACK_WINDOW, PREDICT_WINDOW,
                            CLIP, TRAIN_RATIO, SEED)
    val_ds = KlineDataset(DATA_PATH, 'val', LOOKBACK_WINDOW, PREDICT_WINDOW,
                          CLIP, TRAIN_RATIO, SEED + 1)

    train_loader = DataLoader(train_ds, batch_size=TOK_BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=TOK_BATCH_SIZE, shuffle=False,
                            num_workers=0, drop_last=False)

    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=TOK_LR,
                                  weight_decay=TOK_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=TOK_LR,
        steps_per_epoch=len(train_loader), epochs=TOK_EPOCHS,
        pct_start=0.05, div_factor=10
    )

    best_val_loss = float('inf')
    save_path = os.path.join(SAVE_DIR, "tokenizer", "best_model")
    os.makedirs(save_path, exist_ok=True)
    global_step = 0
    t0 = time.time()

    for epoch in range(TOK_EPOCHS):
        epoch_t0 = time.time()
        tokenizer.train()
        train_ds.set_epoch_seed(epoch * 10000)

        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_bsq = 0.0
        epoch_steps = 0

        for batch_x, _ in train_loader:
            batch_x = batch_x.to(DEVICE)

            (z_pre, z), bsq_loss, _, _ = tokenizer(batch_x)

            recon_loss_pre = F.mse_loss(z_pre, batch_x)
            recon_loss_all = F.mse_loss(z, batch_x)
            recon_loss = recon_loss_pre + recon_loss_all
            loss = (recon_loss + bsq_loss) / 2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_bsq += bsq_loss.item()
            epoch_steps += 1
            global_step += 1

            # Log step-level metrics
            wandb.log({
                "tokenizer/train_loss": loss.item(),
                "tokenizer/recon_loss": recon_loss.item(),
                "tokenizer/recon_loss_pre": recon_loss_pre.item(),
                "tokenizer/recon_loss_all": recon_loss_all.item(),
                "tokenizer/bsq_loss": bsq_loss.item(),
                "tokenizer/lr": optimizer.param_groups[0]['lr'],
                "tokenizer/step": global_step,
            }, step=global_step)

            if global_step % TOK_LOG_INTERVAL == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"    [Epoch {epoch+1}/{TOK_EPOCHS}, Step {global_step}] "
                      f"loss={loss.item():.4f}  recon={recon_loss.item():.4f}  "
                      f"bsq={bsq_loss.item():.4f}  lr={lr:.6f}")

        # Validation
        tokenizer.eval()
        val_loss_sum, val_count = 0.0, 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(DEVICE)
                (_, z), _, _, _ = tokenizer(batch_x)
                val_loss_sum += F.mse_loss(z, batch_x).item() * batch_x.size(0)
                val_count += batch_x.size(0)

        avg_val_loss = val_loss_sum / val_count if val_count > 0 else 0
        avg_train_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0

        # Log epoch-level metrics
        wandb.log({
            "tokenizer/epoch": epoch + 1,
            "tokenizer/epoch_train_loss": avg_train_loss,
            "tokenizer/epoch_recon_loss": epoch_recon / epoch_steps,
            "tokenizer/epoch_bsq_loss": epoch_bsq / epoch_steps,
            "tokenizer/epoch_val_loss": avg_val_loss,
            "tokenizer/best_val_loss": min(best_val_loss, avg_val_loss),
        }, step=global_step)

        print(f"  --- Epoch {epoch+1}/{TOK_EPOCHS} ---  "
              f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f}  "
              f"time={fmt_time(time.time() - epoch_t0)}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            tokenizer.save_pretrained(save_path)
            print(f"  ** Saved best tokenizer (val_loss={best_val_loss:.4f})")

    total_time = time.time() - t0
    print(f"\n  Stage 1 complete. Best val_loss={best_val_loss:.4f}  "
          f"Total time: {fmt_time(total_time)}")
    print(f"  Saved to: {save_path}")

    return save_path, global_step


# ═══════════════════════════════════════════════════════════════════════
# Stage 2: Predictor Pretraining
# ═══════════════════════════════════════════════════════════════════════

def train_predictor(tokenizer_path, step_offset=0):
    print("\n" + "=" * 64)
    print("  STAGE 2: Predictor Pretraining (Kronos-mini)")
    print("=" * 64)

    # Load frozen tokenizer
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    tokenizer.eval().to(DEVICE)
    for p in tokenizer.parameters():
        p.requires_grad = False
    print(f"  Loaded tokenizer from {tokenizer_path}")

    # Initialize predictor from scratch
    model = Kronos(
        s1_bits=PRED_S1_BITS,
        s2_bits=PRED_S2_BITS,
        n_layers=PRED_N_LAYERS,
        d_model=PRED_D_MODEL,
        n_heads=PRED_N_HEADS,
        ff_dim=PRED_FF_DIM,
        ffn_dropout_p=PRED_FFN_DROPOUT,
        attn_dropout_p=PRED_ATTN_DROPOUT,
        resid_dropout_p=PRED_RESID_DROPOUT,
        token_dropout_p=PRED_TOKEN_DROPOUT,
        learn_te=PRED_LEARN_TE,
    ).to(DEVICE)

    total, trainable = count_params(model)
    print(f"  Predictor params: {fmt_params(total)} (trainable: {fmt_params(trainable)})")

    # Data
    print("  Loading data...")
    train_ds = KlineDataset(DATA_PATH, 'train', LOOKBACK_WINDOW, PREDICT_WINDOW,
                            CLIP, TRAIN_RATIO, SEED)
    val_ds = KlineDataset(DATA_PATH, 'val', LOOKBACK_WINDOW, PREDICT_WINDOW,
                          CLIP, TRAIN_RATIO, SEED + 1)

    train_loader = DataLoader(train_ds, batch_size=PRED_BATCH_SIZE, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=PRED_BATCH_SIZE, shuffle=False,
                            num_workers=0, drop_last=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=PRED_LR,
        betas=(PRED_BETA1, PRED_BETA2),
        weight_decay=PRED_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=PRED_LR,
        steps_per_epoch=len(train_loader), epochs=PRED_EPOCHS,
        pct_start=0.05, div_factor=10
    )

    best_val_loss = float('inf')
    save_path = os.path.join(SAVE_DIR, "predictor", "best_model")
    os.makedirs(save_path, exist_ok=True)
    global_step = 0
    t0 = time.time()

    for epoch in range(PRED_EPOCHS):
        epoch_t0 = time.time()
        model.train()
        train_ds.set_epoch_seed(epoch * 10000)

        epoch_loss = 0.0
        epoch_s1_loss = 0.0
        epoch_s2_loss = 0.0
        epoch_steps = 0

        for batch_x, batch_stamp in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_stamp = batch_stamp.to(DEVICE)

            # Tokenize on-the-fly (frozen tokenizer)
            with torch.no_grad():
                token_s1, token_s2 = tokenizer.encode(batch_x, half=True)

            # Autoregressive: input is [:-1], target is [1:]
            s1_in, s2_in = token_s1[:, :-1], token_s2[:, :-1]
            s1_tgt, s2_tgt = token_s1[:, 1:], token_s2[:, 1:]
            stamp_in = batch_stamp[:, :-1, :]

            # Forward with teacher forcing
            s1_logits, s2_logits = model(
                s1_in, s2_in, stamp_in,
                use_teacher_forcing=True, s1_targets=s1_tgt
            )

            loss, s1_loss, s2_loss = model.head.compute_loss(
                s1_logits, s2_logits, s1_tgt, s2_tgt
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_s1_loss += s1_loss.item()
            epoch_s2_loss += s2_loss.item()
            epoch_steps += 1
            global_step += 1

            # Log step-level metrics
            abs_step = step_offset + global_step
            wandb.log({
                "predictor/train_loss": loss.item(),
                "predictor/s1_loss": s1_loss.item(),
                "predictor/s2_loss": s2_loss.item(),
                "predictor/lr": optimizer.param_groups[0]['lr'],
                "predictor/step": global_step,
            }, step=abs_step)

            if global_step % PRED_LOG_INTERVAL == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"    [Epoch {epoch+1}/{PRED_EPOCHS}, Step {global_step}] "
                      f"loss={loss.item():.4f}  s1={s1_loss.item():.4f}  "
                      f"s2={s2_loss.item():.4f}  lr={lr:.6f}")

        # Validation
        model.eval()
        val_loss_sum, val_s1_sum, val_s2_sum, val_batches = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch_x, batch_stamp in val_loader:
                batch_x = batch_x.to(DEVICE)
                batch_stamp = batch_stamp.to(DEVICE)

                token_s1, token_s2 = tokenizer.encode(batch_x, half=True)
                s1_in, s2_in = token_s1[:, :-1], token_s2[:, :-1]
                s1_tgt, s2_tgt = token_s1[:, 1:], token_s2[:, 1:]
                stamp_in = batch_stamp[:, :-1, :]

                s1_logits, s2_logits = model(s1_in, s2_in, stamp_in)
                val_loss, val_s1, val_s2 = model.head.compute_loss(
                    s1_logits, s2_logits, s1_tgt, s2_tgt
                )
                val_loss_sum += val_loss.item()
                val_s1_sum += val_s1.item()
                val_s2_sum += val_s2.item()
                val_batches += 1

        avg_val_loss = val_loss_sum / val_batches if val_batches > 0 else 0
        avg_val_s1 = val_s1_sum / val_batches if val_batches > 0 else 0
        avg_val_s2 = val_s2_sum / val_batches if val_batches > 0 else 0
        avg_train_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        avg_s1 = epoch_s1_loss / epoch_steps if epoch_steps > 0 else 0
        avg_s2 = epoch_s2_loss / epoch_steps if epoch_steps > 0 else 0

        # Log epoch-level metrics
        abs_step = step_offset + global_step
        wandb.log({
            "predictor/epoch": epoch + 1,
            "predictor/epoch_train_loss": avg_train_loss,
            "predictor/epoch_s1_loss": avg_s1,
            "predictor/epoch_s2_loss": avg_s2,
            "predictor/epoch_val_loss": avg_val_loss,
            "predictor/epoch_val_s1_loss": avg_val_s1,
            "predictor/epoch_val_s2_loss": avg_val_s2,
            "predictor/best_val_loss": min(best_val_loss, avg_val_loss),
        }, step=abs_step)

        print(f"  --- Epoch {epoch+1}/{PRED_EPOCHS} ---  "
              f"train_loss={avg_train_loss:.4f} (s1={avg_s1:.4f}, s2={avg_s2:.4f})  "
              f"val_loss={avg_val_loss:.4f}  time={fmt_time(time.time() - epoch_t0)}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(save_path)
            print(f"  ** Saved best predictor (val_loss={best_val_loss:.4f})")

    total_time = time.time() - t0
    print(f"\n  Stage 2 complete. Best val_loss={best_val_loss:.4f}  "
          f"Total time: {fmt_time(total_time)}")
    print(f"  Saved to: {save_path}")

    return save_path


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         Kronos-mini Pretraining from Scratch                ║")
    print("║         CPU-only · XSHG 5-min K-line data                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Device:  {DEVICE}")
    print(f"  Data:    {DATA_PATH}")
    print(f"  Save to: {SAVE_DIR}")
    print(f"  Seed:    {SEED}")

    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Save config for reproducibility
    config = {
        "tokenizer": {
            "d_in": TOK_D_IN, "d_model": TOK_D_MODEL, "n_heads": TOK_N_HEADS,
            "ff_dim": TOK_FF_DIM, "n_enc_layers": TOK_N_ENC_LAYERS,
            "n_dec_layers": TOK_N_DEC_LAYERS, "s1_bits": TOK_S1_BITS,
            "s2_bits": TOK_S2_BITS, "epochs": TOK_EPOCHS, "lr": TOK_LR,
            "batch_size": TOK_BATCH_SIZE,
        },
        "predictor": {
            "s1_bits": PRED_S1_BITS, "s2_bits": PRED_S2_BITS,
            "n_layers": PRED_N_LAYERS, "d_model": PRED_D_MODEL,
            "n_heads": PRED_N_HEADS, "ff_dim": PRED_FF_DIM,
            "epochs": PRED_EPOCHS, "lr": PRED_LR,
            "batch_size": PRED_BATCH_SIZE,
        },
        "data": {
            "path": DATA_PATH, "lookback": LOOKBACK_WINDOW,
            "predict": PREDICT_WINDOW, "clip": CLIP,
            "train_ratio": TRAIN_RATIO,
        },
    }
    with open(os.path.join(SAVE_DIR, "pretrain_config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    # ── Initialize Wandb ──────────────────────────────────────────────
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name="kronos-mini-pretrain",
        config=config,
        tags=["pretrain", "kronos-mini", "cpu", "XSHG-5min"],
    )

    # ── Stage 1: Tokenizer ────────────────────────────────────────────
    tokenizer_path, tok_steps = train_tokenizer()

    # ── Stage 2: Predictor ────────────────────────────────────────────
    predictor_path = train_predictor(tokenizer_path, step_offset=tok_steps)

    # ── Finish Wandb ──────────────────────────────────────────────────
    wandb.finish()

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  PRETRAINING COMPLETE")
    print("=" * 64)
    print(f"  Tokenizer: {tokenizer_path}")
    print(f"  Predictor: {predictor_path}")
    print()
    print("  To use the pretrained model:")
    print("    from model import Kronos, KronosTokenizer, KronosPredictor")
    print(f'    tokenizer = KronosTokenizer.from_pretrained("{tokenizer_path}")')
    print(f'    model = Kronos.from_pretrained("{predictor_path}")')
    print("    predictor = KronosPredictor(model, tokenizer, max_context=512)")
    print("=" * 64)


if __name__ == "__main__":
    main()
