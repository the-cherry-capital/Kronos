#!/bin/bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════
# 1/3 — Environment Setup
# ═══════════════════════════════════════════════════════════════════════
#
# Clones repo, creates venv, installs all dependencies.
#
# Usage:
#   bash scripts/setup_env.sh
#
# Env vars:
#   REPO_URL    — git clone URL
#   BRANCH      — branch to checkout
#   INSTALL_DIR — where to clone (default: ./Kronos)
# ═══════════════════════════════════════════════════════════════════════

REPO_URL="${REPO_URL:-https://github.com/the-cherry-capital/Kronos.git}"
BRANCH="${BRANCH:-pretrain-kronos-mini}"
INSTALL_DIR="${INSTALL_DIR:-$(pwd)/Kronos}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║    [1/3] Environment Setup                                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Clone & checkout ─────────────────────────────────────────────────
if [ ! -d "$INSTALL_DIR/.git" ]; then
    echo "Cloning repo..."
    git clone "$REPO_URL" "$INSTALL_DIR"
else
    echo "Repo exists, pulling latest..."
    git -C "$INSTALL_DIR" fetch origin
fi
cd "$INSTALL_DIR"
git checkout "$BRANCH"
git pull origin "$BRANCH" || true

# ── Virtual environment ──────────────────────────────────────────────
echo "Setting up Python venv..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip -q

# Install PyTorch: use CUDA wheels when a GPU is detected, CPU otherwise.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "  GPU detected — installing PyTorch with CUDA support"
    pip install torch --index-url https://download.pytorch.org/whl/cu130 -q
else
    echo "  No GPU detected — installing CPU PyTorch"
    pip install torch -q
fi

pip install -r requirements.txt -q

echo ""
echo "  Python: $(python3 --version)"
echo "  Torch:  $(python3 -c 'import torch; print(torch.__version__)')"
echo "  CUDA:   $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# ── Wandb check ──────────────────────────────────────────────────────
if python3 -c "import wandb; assert wandb.api.api_key" 2>/dev/null; then
    echo "  wandb:  logged in"
else
    echo "  wandb:  not logged in (run 'wandb login' to enable)"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Environment ready at: $INSTALL_DIR"
echo "  Activate with: cd $INSTALL_DIR && source .venv/bin/activate"
echo ""
echo "  Next: bash scripts/prepare_data.sh"
echo "════════════════════════════════════════════════════════════════"
