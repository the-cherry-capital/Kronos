# Kronos-mini Pretraining from Scratch

Reproduce the pretraining of [Kronos-mini](https://huggingface.co/NeoQuasar/Kronos-mini) (4.1M params) — the first open-source foundation model for financial K-lines — with full distributed training support.

Based on the [Kronos](https://github.com/shiyu-coder/Kronos) framework (AAAI 2026). Two-stage pretraining: BSQ tokenizer + autoregressive predictor on OHLCV candlestick data.

## Quick Start

### One shot — download data, setup env, and train:

```bash
bash setup_and_train.sh
```

### Or step by step:

**Step 1: Download data**

```bash
bash setup_data.sh
```

**Step 2: Train (skips download since data already exists)**

```bash
cd Kronos && SKIP_DOWNLOAD=1 bash setup_and_train.sh
```

### Remote server — curl and go:

```bash
# Download & preprocess data
curl -sL https://raw.githubusercontent.com/the-cherry-capital/Kronos/pretrain-kronos-mini/setup_data.sh | bash

# Train
cd Kronos && SKIP_DOWNLOAD=1 NUM_GPUS=4 bash setup_and_train.sh
```

## Multi-GPU / Multi-Node

```bash
# Single node, 4 GPUs
NUM_GPUS=4 bash setup_and_train.sh

# Single node, 8 GPUs, bigger batches
NUM_GPUS=8 BATCH_SIZE=128 bash setup_and_train.sh

# Multi-node (2 nodes x 8 GPUs) — run on each node:
# Node 0 (master)
NUM_GPUS=8 NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash setup_and_train.sh

# Node 1
NUM_GPUS=8 NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 bash setup_and_train.sh
```

## Configuration

All configurable via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_GPUS` | `0` | Number of GPUs per node (0 = single process CPU/auto) |
| `NNODES` | `1` | Number of nodes |
| `NODE_RANK` | `0` | This node's rank |
| `MASTER_ADDR` | `127.0.0.1` | Master node address |
| `MASTER_PORT` | `29500` | Master node port |
| `BATCH_SIZE` | `64` | Per-GPU batch size |
| `TOK_EPOCHS` | `3` | Tokenizer training epochs |
| `PRED_EPOCHS` | `5` | Predictor training epochs |
| `MAX_SAMPLES` | `1000000` | Cap on training samples |
| `NUM_WORKERS` | `4` | Dataloader workers |
| `SKIP_DOWNLOAD` | `0` | Set to `1` to skip data download |

Or use `pretrain_scaled.py` directly with argparse:

```bash
torchrun --standalone --nproc_per_node=4 pretrain_scaled.py \
    --tok_epochs 5 --pred_epochs 10 --tok_batch_size 128 --pred_batch_size 128
```

## What's in the Box

| File | Description |
|------|-------------|
| `setup_and_train.sh` | One-shot: clone, venv, deps, download data, train |
| `setup_data.sh` | One-shot: clone, venv, deps, download data only |
| `download_data.py` | Downloads ~200 tickers daily OHLCV (2004-2025) via yfinance |
| `pretrain_scaled.py` | Full DDP pretraining script (multi-node, multi-GPU, wandb) |
| `pretrain_mini.py` | Minimal single-process pretraining on example CSV |

## Architecture

Uses the official Kronos-mini config from HuggingFace:

**Tokenizer** (Kronos-Tokenizer-2k, 3.96M params):
- Encoder-decoder transformer with Binary Spherical Quantization
- d_model=256, n_heads=4, ff_dim=512, 4 enc/dec layers
- s1_bits=10, s2_bits=10 (hierarchical token scheme)

**Predictor** (Kronos-mini, 4.11M params):
- Decoder-only autoregressive transformer with RoPE
- d_model=256, n_heads=4, ff_dim=512, 4 layers
- Hierarchical embedding + dependency-aware layer + dual head

## Data

`download_data.py` fetches daily OHLCV from ~200 diverse tickers:
- US large/mid-cap across all sectors (tech, finance, healthcare, energy, etc.)
- Sector & broad market ETFs (SPY, QQQ, XLF, EEM, etc.)
- ~987K total rows, 2004-2025

## Wandb

Losses are logged to [wandb](https://wandb.ai). Login before training:

```bash
source .venv/bin/activate && wandb login
```

Or disable with `--no_wandb` flag.

## Citation

Based on the Kronos paper (AAAI 2026):

```
@misc{shi2025kronos,
      title={Kronos: A Foundation Model for the Language of Financial Markets},
      author={Yu Shi and Zongliang Fu and Shuo Chen and Bohan Zhao and Wei Xu and Changshui Zhang and Jian Li},
      year={2025},
      eprint={2508.02739},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST},
      url={https://arxiv.org/abs/2508.02739},
}
```

## License

[MIT License](./LICENSE)
