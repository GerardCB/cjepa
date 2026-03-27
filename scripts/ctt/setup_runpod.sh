#!/bin/bash
# ============================================================================
# CTT-JEPA: RunPod / GPU Server Setup Script
# Run this once after spinning up a pod.
# Tested on: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# ============================================================================
set -e

echo "========================================"
echo "  CTT-JEPA Environment Setup"
echo "========================================"

# 1. Install Python dependencies
echo "[1/6] Installing Python dependencies..."
pip install -q hydra-core omegaconf einops loguru wandb tqdm seaborn

# 2. Install ALOE-specific dependencies
echo "[2/6] Installing ALOE VQA dependencies..."
pip install -q 'torchmetrics<1.0' pycocotools webdataset
# torchmetrics<1.0 required for nerv compatibility (compute_on_step arg)

# 3. Install third-party libraries
echo "[3/6] Installing third-party libraries..."
cd src/third_party

if [ ! -d "stable-pretraining" ]; then
    git clone --quiet https://github.com/galilai-group/stable-pretraining.git
    cd stable-pretraining && git checkout 92b5841 && pip install -q -e . && cd ..
else
    echo "  stable-pretraining already installed"
fi

if [ ! -d "stable-worldmodel" ]; then
    git clone --quiet https://github.com/galilai-group/stable-worldmodel.git
    pip install -q -e stable-worldmodel
else
    echo "  stable-worldmodel already installed"
fi

if [ ! -d "nerv" ]; then
    git clone --quiet https://github.com/Wuziyi616/nerv.git
    cd nerv && git checkout v0.1.0 && pip install -q --ignore-installed blinker && pip install -q -e . && cd ..
else
    echo "  nerv already installed"
fi

cd ../..

# 4. Fix torchcodec imports (crashes on many CUDA versions)
echo "[4/6] Patching torchcodec imports..."
pip uninstall torchcodec -y 2>/dev/null || true
PYTHONPATH=$(pwd) python scripts/ctt/fix_torchcodec.py

# 5. Download slot embeddings
echo "[5/6] Downloading VideoSAUR slot embeddings (~9 GB)..."
mkdir -p data
if [ ! -f "data/clevrer_videosaur_slots.pkl" ]; then
    pip install -q huggingface_hub
    python -c "
from huggingface_hub import hf_hub_download
import os
hf_hub_download(repo_id='HazelNam/CJEPA', filename='clevrer_videosaur_slots.pkl',
                local_dir=os.path.join(os.getcwd(), 'data'))
print('Download complete.')
"
else
    echo "  Slot embeddings already downloaded"
fi

# 6. Verify installation
echo "[6/6] Verifying installation..."
PYTHONPATH=$(pwd) python -c "
import torch
from src.cjepa_predictor import MaskedSlotPredictor
from src.ctt_losses import ctt_invariance_loss, ctt_sufficiency_loss
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
print('  All imports OK!')
"

echo ""
echo "========================================"
echo "  Setup complete!"
echo "  Next steps:"
echo "    1. bash scripts/ctt/train_worldmodels.sh"
echo "    2. bash scripts/ctt/train_aloe.sh"
echo "========================================"

