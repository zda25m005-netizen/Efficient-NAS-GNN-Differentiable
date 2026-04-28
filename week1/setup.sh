#!/bin/bash
# Week 1 Setup Script — run this once in your VS Code terminal
# Usage: bash setup.sh

set -e
echo ""
echo "=================================================="
echo "  Week 1: Neural Architecture Search — Setup"
echo "=================================================="
echo ""

# ── 1. Check Python version ───────────────────────────────────────────────────
echo "→ Checking Python..."
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
echo "  Found Python $PY_VERSION"

# PyTorch supports Python 3.9–3.12 only (3.13/3.14 not yet supported)
if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -gt 12 ]; then
  echo ""
  echo "  ⚠️  Python $PY_VERSION detected — PyTorch requires Python ≤ 3.12"
  echo "  Installing Python 3.12 via Homebrew and using it for this project..."
  echo ""

  # Install Python 3.12 if not already present
  if ! command -v python3.12 &>/dev/null; then
    echo "  → brew install python@3.12  (this may take a minute)..."
    brew install python@3.12
  fi

  PYTHON="python3.12"
  echo "  → Using: $($PYTHON --version)"
else
  PYTHON="python3"
fi

# ── 2. Create virtual environment ────────────────────────────────────────────
echo ""
echo "→ Creating virtual environment (.venv) with $PYTHON..."
$PYTHON -m venv .venv
source .venv/bin/activate
echo "  Activated: $(python --version)"

# ── 3. Install dependencies ───────────────────────────────────────────────────
echo ""
echo "→ Installing dependencies..."
echo "  (PyTorch CPU build — ~700MB, may take 2-5 minutes)"
echo ""

python -m pip install --upgrade pip -q

# CPU-only torch (works on any Mac including Apple Silicon)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Scientific stack
python -m pip install numpy scipy matplotlib

echo ""
echo "✓ Dependencies installed."

# ── 4. Verify installation ───────────────────────────────────────────────────
echo ""
echo "→ Verifying installation..."
python -c "
import torch, torchvision, numpy, scipy, matplotlib
print(f'  torch       {torch.__version__}')
print(f'  torchvision {torchvision.__version__}')
print(f'  numpy       {numpy.__version__}')
print(f'  scipy       {scipy.__version__}')
print(f'  matplotlib  {matplotlib.__version__}')
print()
print('  All packages OK ✓')
"

# ── 5. Run smoke test ────────────────────────────────────────────────────────
echo ""
echo "→ Running smoke test (validates all 5 Week 1 deliverables)..."
echo ""
python smoke_test.py --device cpu

# ── 6. Quick model check ─────────────────────────────────────────────────────
echo ""
echo "→ Quick ResNet-20 forward pass check..."
python -c "
import torch, sys; sys.path.insert(0, '.')
from models.resnet import resnet20
model = resnet20()
x = torch.randn(4, 3, 32, 32)
y = model(x)
print(f'  ResNet-20: {model.count_parameters():,} params | output {y.shape} ✓')
"

echo ""
echo "=================================================="
echo "  SETUP COMPLETE — Week 1 is ready!"
echo "=================================================="
echo ""
echo "  ⚡ IMPORTANT: Virtual environment is active for this terminal."
echo "  If you open a new terminal, re-activate it with:"
echo "     source .venv/bin/activate"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Train ResNet-20 to 94% accuracy:"
echo "     python train_resnet20.py --device cpu"
echo ""
echo "  2. Run correlation analysis (proxies vs. accuracy):"
echo "     python experiments/correlation_analysis.py --n_archs 100"
echo ""
echo "  3. Quick smoke test anytime:"
echo "     python smoke_test.py"
echo ""
