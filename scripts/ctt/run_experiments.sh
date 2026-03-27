#!/bin/bash
# ============================================================================
# CTT-JEPA: Run Baseline + CTT Experiments
# Expected time: ~3.5 hours on 1x H100
# ============================================================================
set -e

export PYTHONPATH=$(pwd)
DATA="./data/clevrer_videosaur_slots.pkl"
TRAIN_CMD="python src/train/train_causalwm_from_clevrer_slot.py"
COMMON_ARGS="embedding_dir=$DATA wandb.enable=false rollout.save_rollout=true"

echo "========================================"
echo "  CTT-JEPA Experiments"
echo "========================================"
echo ""

# ── Experiment 1: C-JEPA Baseline ──
echo "[1/4] Training C-JEPA Baseline (30 epochs)..."
$TRAIN_CMD $COMMON_ARGS \
    output_model_name=baseline_videosaur \
    ctt_inv_weight=0.0 \
    ctt_suf_weight=0.0

echo "[1/4] ✓ Baseline training + rollout complete"
echo ""

# ── Experiment 2: CTT-JEPA (Full) ──
echo "[2/4] Training CTT-JEPA Full (30 epochs, CTT from epoch 10)..."
$TRAIN_CMD $COMMON_ARGS \
    output_model_name=ctt_full_videosaur \
    ctt_inv_weight=0.2 \
    ctt_suf_weight=0.1 \
    ctt_start_epoch=10

echo "[2/4] ✓ CTT-JEPA training + rollout complete"
echo ""

# ── Experiment 3: ALOE VQA (Baseline) ──
echo "[3/4] Training ALOE VQA on baseline rollout..."
BASELINE_ROLLOUT=$(find . -name "rollout_*baseline_videosaur*" -path "*/data/*" 2>/dev/null | head -1)
if [ -z "$BASELINE_ROLLOUT" ]; then
    BASELINE_ROLLOUT=$(find . -name "rollout_*baseline*" 2>/dev/null | head -1)
fi

if [ -n "$BASELINE_ROLLOUT" ]; then
    echo "  Using rollout: $BASELINE_ROLLOUT"
    bash scripts/clevrer/train_aloe.sh "$BASELINE_ROLLOUT" baseline 2>&1 | tee logs/aloe_baseline.log
    echo "[3/4] ✓ Baseline ALOE training complete"
else
    echo "[3/4] ⚠ Baseline rollout not found — skipping ALOE"
    echo "  You may need to re-run Step 1 with rollout.save_rollout=true"
fi
echo ""

# ── Experiment 4: ALOE VQA (CTT) ──
echo "[4/4] Training ALOE VQA on CTT rollout..."
CTT_ROLLOUT=$(find . -name "rollout_*ctt_full_videosaur*" -path "*/data/*" 2>/dev/null | head -1)
if [ -z "$CTT_ROLLOUT" ]; then
    CTT_ROLLOUT=$(find . -name "rollout_*ctt_full*" 2>/dev/null | head -1)
fi

if [ -n "$CTT_ROLLOUT" ]; then
    echo "  Using rollout: $CTT_ROLLOUT"
    bash scripts/clevrer/train_aloe.sh "$CTT_ROLLOUT" ctt_full 2>&1 | tee logs/aloe_ctt.log
    echo "[4/4] ✓ CTT ALOE training complete"
else
    echo "[4/4] ⚠ CTT rollout not found — skipping ALOE"
fi

echo ""
echo "========================================"
echo "  All experiments complete!"
echo "  Check logs/ for detailed output."
echo "  Results:"

if [ -f "logs/aloe_baseline.log" ]; then
    echo "  Baseline: $(grep -i 'accuracy\|vqa' logs/aloe_baseline.log | tail -3)"
fi
if [ -f "logs/aloe_ctt.log" ]; then
    echo "  CTT-JEPA: $(grep -i 'accuracy\|vqa' logs/aloe_ctt.log | tail -3)"
fi

echo "========================================"
