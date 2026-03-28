#!/bin/bash
# CTT-JEPA: Train both world models (baseline + CTT) on RunPod H100
# Usage: bash scripts/ctt/train_worldmodels.sh
set -e
cd /workspace/cjepa
export PYTHONPATH=/workspace/cjepa
mkdir -p logs checkpoints

DATA="/workspace/cjepa/data/clevrer_videosaur_slots.pkl"
TRAIN="python src/train/train_causalwm_from_clevrer_slot.py"

echo "========================================"
echo "  CTT-JEPA World Model Training"
echo "  Started: $(date)"
echo "========================================"

# 1. Baseline C-JEPA (30 epochs, no CTT losses)
echo "[1/2] Training C-JEPA Baseline..."
$TRAIN embedding_dir=$DATA wandb.enable=false batch_size=2048 \
    trainer.max_epochs=30 rollout.save_rollout=true \
    output_model_name=baseline_videosaur \
    ctt_inv_weight=0.0 ctt_suf_weight=0.0 \
    cache_dir=/workspace/cjepa/checkpoints \
    2>&1 | tee logs/train_baseline.log
echo "[1/2] Baseline DONE at $(date)"

# 2. CTT-JEPA Full (30 epochs, CTT losses activated at epoch 10)
echo "[2/2] Training CTT-JEPA Full..."
$TRAIN embedding_dir=$DATA wandb.enable=false batch_size=2048 \
    trainer.max_epochs=30 rollout.save_rollout=true \
    output_model_name=ctt_full_videosaur \
    ctt_inv_weight=0.2 ctt_suf_weight=0.1 ctt_start_epoch=10 \
    cache_dir=/workspace/cjepa/checkpoints \
    2>&1 | tee logs/train_ctt.log
echo "[2/2] CTT DONE at $(date)"

echo "========================================"
echo "  World model training complete!"
echo "  Finished: $(date)"
echo "========================================"
echo ""
echo "Checkpoints saved to: /workspace/cjepa/checkpoints/"
echo "Rollouts saved to: /workspace/cjepa/data/"
echo ""
echo "Next step: bash scripts/ctt/train_aloe.sh"
