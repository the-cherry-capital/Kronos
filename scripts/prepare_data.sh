#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════
# 2/3 — Data Download & Preprocess
# ═══════════════════════════════════════════════════════════════════════
#
# Downloads hourly Yahoo Finance data for a curated set of popular tickers
# and preprocesses it into Kronos CSV format.
#
# Prerequisite: run scripts/setup_env.sh first.
#
# Usage:
#   bash scripts/prepare_data.sh
#
# Env vars:
#   INSTALL_DIR          — repo location (default: ./Kronos)
#   PRETRAIN_TICKERS            — optional comma-separated ticker override
#   PRETRAIN_FREQUENCY          — bar size (default: 60m)
#   PRETRAIN_LOOKBACK           — provider lookback window (default: 730d)
#   PRETRAIN_TOTAL_ROWS         — target total rows across all tickers (default: 100000)
#   PRETRAIN_MIN_ROWS_PER_TICKER — minimum usable rows per ticker (default: 1000)
# ═══════════════════════════════════════════════════════════════════════

INSTALL_DIR="${INSTALL_DIR:-$(pwd)}"

# Auto-detect: if we're inside the repo already, use cwd
if [ ! -f "$INSTALL_DIR/download_data.py" ]; then
    # Try parent in case scripts/ is cwd
    INSTALL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║    [2/3] Data Download & Preprocess                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"

cd "$INSTALL_DIR"
source .venv/bin/activate

echo "Downloading hourly data from popular tickers..."
echo "  Frequency:   ${PRETRAIN_FREQUENCY:-60m}"
echo "  Lookback:    ${PRETRAIN_LOOKBACK:-730d}"
echo "  Target rows: ${PRETRAIN_TOTAL_ROWS:-100000}"
if [ -n "${PRETRAIN_TICKERS:-}" ]; then
    echo "  Tickers:     ${PRETRAIN_TICKERS}"
fi
python3 download_data.py

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

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Data ready!"
echo "  Location:   $DATA_DIR"
echo "  Tickers:    $N_FILES"
echo "  Total rows: $TOTAL_ROWS"
echo ""
echo "  Next: bash scripts/launch_training.sh"
echo "════════════════════════════════════════════════════════════════"
