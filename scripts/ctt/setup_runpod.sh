#!/bin/bash
# ============================================================================
# CTT-JEPA: RunPod / GPU Server Setup Script
# Run this once after spinning up a pod.
# ============================================================================
set -e

echo "========================================"
echo "  CTT-JEPA Environment Setup"
echo "========================================"

# 1. Install Python dependencies
echo "[1/4] Installing Python dependencies..."
pip install -q hydra-core omegaconf einops loguru wandb tqdm seaborn

# 2. Install third-party libraries
echo "[2/4] Installing third-party libraries..."
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
    cd nerv && git checkout v0.1.0 && pip install -q -e . && cd ..
else
    echo "  nerv already installed"
fi

cd ../..

# 3. Download slot embeddings
echo "[3/4] Downloading VideoSAUR slot embeddings (~9 GB)..."
if [ ! -f "data/clevrer_videosaur_slots.pkl" ]; then
    pip install -q huggingface_hub
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='HazelNam/CJEPA', filename='clevrer_videosaur_slots.pkl', local_dir='./data')
print('Download complete.')
"
else
    echo "  Slot embeddings already downloaded"
fi

# 4. Verify installation
echo "[4/4] Verifying installation..."
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
echo "  Run:  bash scripts/ctt/run_experiments.sh"
echo "========================================"
