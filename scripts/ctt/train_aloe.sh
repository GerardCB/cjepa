#!/bin/bash
# CTT-JEPA: Train ALOE VQA models on both rollouts
# Prerequisites: Run train_worldmodels.sh first (produces rollout .pkl files)
# Usage: bash scripts/ctt/train_aloe.sh
set -e
cd /workspace/cjepa
export PYTHONPATH=/workspace/cjepa
export WANDB_MODE=disabled
mkdir -p logs aloe_output

ALOE="python src/aloe_train.py --task clevrer_vqa \
    --params src/third_party/slotformer/clevrer_vqa/configs/aloe_clevrer_params-rollout.py \
    --fp16 --cudnn"

# IMPORTANT: Before first run, you may need to fix torchcodec imports:
#   python scripts/ctt/fix_torchcodec.py
# And install: pip install 'torchmetrics<1.0' pycocotools webdataset

echo "========================================"
echo "  ALOE VQA Training"
echo "  Started: $(date)"
echo "========================================"

# Rename rollouts for clarity if they have default names
BASELINE_ROLLOUT="/workspace/cjepa/data/rollout_baseline.pkl"
CTT_ROLLOUT="/workspace/cjepa/data/rollout_ctt.pkl"

# Check that rollouts exist
if [ ! -f "$BASELINE_ROLLOUT" ]; then
    # Try the default name and rename
    DEFAULT_ROLLOUT="/workspace/cjepa/data/rollout_clevrer_videosaur_slots_lr0.0005_mask2.pkl"
    if [ -f "$DEFAULT_ROLLOUT" ]; then
        echo "WARNING: Only found default rollout. Run both world models first."
        echo "Copying default rollout as baseline..."
        cp "$DEFAULT_ROLLOUT" "$BASELINE_ROLLOUT"
    else
        echo "ERROR: No rollout found. Run train_worldmodels.sh first."
        exit 1
    fi
fi

if [ ! -f "$CTT_ROLLOUT" ]; then
    echo "ERROR: CTT rollout not found at $CTT_ROLLOUT"
    echo "Run train_worldmodels.sh first, then rename rollouts."
    exit 1
fi

# 1. ALOE on Baseline rollout
echo "[1/2] Training ALOE on Baseline rollout..."
$ALOE --exp_name aloe_baseline \
    --out_dir /workspace/cjepa/aloe_output \
    --slot_root_override "$BASELINE_ROLLOUT" \
    2>&1 | tee logs/aloe_baseline.log
echo "[1/2] Baseline ALOE DONE at $(date)"

# 2. ALOE on CTT rollout
echo "[2/2] Training ALOE on CTT rollout..."
$ALOE --exp_name aloe_ctt \
    --out_dir /workspace/cjepa/aloe_output \
    --slot_root_override "$CTT_ROLLOUT" \
    2>&1 | tee logs/aloe_ctt.log
echo "[2/2] CTT ALOE DONE at $(date)"

echo "========================================"
echo "  ALOE VQA Training Complete!"
echo "  Finished: $(date)"
echo "========================================"
echo ""
echo "Check eval results with:"
echo "  grep 'Eval epoch' logs/aloe_baseline.log"
echo "  grep 'Eval epoch' logs/aloe_ctt.log"
