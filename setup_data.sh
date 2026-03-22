#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════
# Kronos Pretraining Data — One-Shot Download & Preprocess
# ═══════════════════════════════════════════════════════════════════════
#
# Downloads ~200 diverse tickers (daily OHLCV, 2004-2025) via yfinance,
# preprocesses into Kronos CSV format, and saves to data/pretrain/.
#
# Usage:
#   bash setup_data.sh
#
# Env vars:
#   INSTALL_DIR   — repo location (default: ./Kronos)
#   REPO_URL      — git clone URL
#   BRANCH        — branch to checkout
# ═══════════════════════════════════════════════════════════════════════

REPO_URL="${REPO_URL:-https://github.com/the-cherry-capital/Kronos.git}"
BRANCH="${BRANCH:-pretrain-kronos-mini}"
INSTALL_DIR="${INSTALL_DIR:-$(pwd)/Kronos}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║    Kronos Data — Download & Preprocess                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── 1. Clone & checkout ──────────────────────────────────────────────
if [ ! -d "$INSTALL_DIR/.git" ]; then
    echo "[1/3] Cloning repo..."
    git clone "$REPO_URL" "$INSTALL_DIR"
else
    echo "[1/3] Repo exists, pulling latest..."
    git -C "$INSTALL_DIR" fetch origin
fi
cd "$INSTALL_DIR"
git checkout "$BRANCH"
git pull origin "$BRANCH" || true

# ── 2. Virtual environment & deps ────────────────────────────────────
echo "[2/3] Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

# ── 3. Download & preprocess ─────────────────────────────────────────
echo "[3/3] Downloading market data..."
python3 download_data.py

echo ""
DATA_DIR="$INSTALL_DIR/data/pretrain"
N_FILES=$(find "$DATA_DIR" -name "*.csv" | wc -l | tr -d ' ')
TOTAL_ROWS=$(python3 -c "
import json, os
m = os.path.join('$DATA_DIR', 'manifest.json')
if os.path.exists(m):
    with open(m) as f: print(json.load(f)['total_rows'])
else:
    print('unknown')
")

echo "════════════════════════════════════════════════════════════════"
echo "  Done!"
echo "  Location:  $DATA_DIR"
echo "  Tickers:   $N_FILES"
echo "  Total rows: $TOTAL_ROWS"
echo ""
echo "  Next: run training with"
echo "    cd $INSTALL_DIR && source .venv/bin/activate"
echo "    SKIP_DOWNLOAD=1 bash setup_and_train.sh"
echo "════════════════════════════════════════════════════════════════"
