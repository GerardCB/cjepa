# CTT-JEPA: Causal Transformation Theory for Joint Embedding Predictive Architectures

> **Extending [C-JEPA](https://github.com/galilai-group/cjepa) with causal inductive biases from Causal Transformation Theory (CTT)**

## Key Idea

C-JEPA learns a causal world model through **masked slot prediction** — masking object slots and predicting them from context. We add two loss terms derived from [Causal Transformation Theory](https://arxiv.org/abs/2501.XXXXX) that enforce the attention-based causal graph to satisfy fundamental causal axioms:

| Loss | CTT Axiom | What it enforces |
|---|---|---|
| **Invariance** | Axiom 6 | Masking a non-interacting slot should *not* degrade predictions for unrelated slots |
| **Sufficiency** | Axiom 4 | A slot's causal neighborhood alone should *suffice* to predict it |

These losses use the transformer's own attention weights as the causal adjacency graph — **no architecture changes** are required. The only modification is two additive loss terms controlled via config flags.

## Results

> 🔬 *Benchmarks running — results will be updated here.*

| Model | Counterfactual VQA | Predictive VQA | Avg |
|---|---|---|---|
| C-JEPA (baseline, VideoSAUR) | — | — | 89.40% |
| **CTT-JEPA (ours)** | — | — | **TBD** |

## What Changed from C-JEPA

Only **3 files** were modified/added:

```
src/ctt_losses.py                              [NEW]  — CTT loss functions
src/train/train_causalwm_from_clevrer_slot.py  [MOD]  — Integrated CTT into compute_loss()
configs/config_train_causal_clevrer_slot.yaml   [MOD]  — Added CTT config flags
```

<details>
<summary><strong>Detailed changes</strong></summary>

### `src/ctt_losses.py`
- `ctt_invariance_loss()` — Runs a second forward pass with an additional slot masked. Penalizes prediction degradation for non-neighbor slots.
- `ctt_sufficiency_loss()` — Masks non-neighbors of a target slot. Penalizes poor prediction from the neighborhood alone.
- `_get_slot_attention()` — Extracts slot-to-slot attention with gradients (unlike the original `@torch.no_grad` `attention_probing()`).

### `compute_loss()` modifications
```python
# Original C-JEPA
total_loss = loss_masked_history + loss_future

# CTT-JEPA (our contribution)
total_loss = loss_masked_history + loss_future
           + ctt_inv_weight * invariance_loss    # 0.0 = disabled
           + ctt_suf_weight * sufficiency_loss   # 0.0 = disabled
```

### Config flags
```yaml
ctt_inv_weight: 0.0        # invariance loss weight (0 = C-JEPA baseline)
ctt_suf_weight: 0.0        # sufficiency loss weight (0 = C-JEPA baseline)
ctt_adj_threshold: 0.15    # attention threshold for neighbor detection
ctt_start_epoch: 10        # phased: CTT losses activate after this epoch
```
</details>

## Quick Start (RunPod / GPU Server)

### One-command setup

```bash
bash scripts/ctt/setup_runpod.sh
```

### Run experiments

```bash
bash scripts/ctt/run_experiments.sh
```

This runs:
1. **Baseline C-JEPA** (30 epochs, `ctt_inv_weight=0, ctt_suf_weight=0`)
2. **CTT-JEPA** (30 epochs, `ctt_inv_weight=0.2, ctt_suf_weight=0.1`, phased from epoch 10)
3. **Rollout** for both models (128 → 160 frames)
4. **ALOE VQA eval** on both rollouts

Estimated time: **~3.5 hours on 1x H100**.

### Manual run

```bash
# Activate environment
conda activate cjepa

# Train CTT-JEPA
PYTHONPATH=$(pwd) python src/train/train_causalwm_from_clevrer_slot.py \
    embedding_dir=./data/clevrer_videosaur_slots.pkl \
    ctt_inv_weight=0.2 ctt_suf_weight=0.1 ctt_start_epoch=10 \
    rollout.save_rollout=true output_model_name=ctt_full
```

## Repository Structure

```
cjepa/
├── src/
│   ├── ctt_losses.py                    ← Our CTT loss functions
│   ├── cjepa_predictor.py               ← MaskedSlotPredictor (unmodified)
│   ├── train/
│   │   └── train_causalwm_from_clevrer_slot.py  ← Modified training loop
│   └── third_party/                     ← Dependencies
├── configs/
│   └── config_train_causal_clevrer_slot.yaml    ← Modified config
├── scripts/
│   ├── ctt/
│   │   ├── setup_runpod.sh              ← One-command environment setup
│   │   └── run_experiments.sh           ← Run baseline + CTT experiments
│   └── clevrer/                         ← Original ALOE eval scripts
└── data/                                ← Slot embeddings (downloaded)
```

## Citation

```bibtex
@article{ctt-jepa2026,
    title={CTT-JEPA: Enforcing Causal Axioms in Joint Embedding Predictive Architectures},
    author={Gerard Calvo Bartra},
    year={2026}
}
```

## Acknowledgments

Built on top of [C-JEPA](https://github.com/galilai-group/cjepa) by the Galilai Group. CTT axioms adapted from Causal Transformation Theory.
