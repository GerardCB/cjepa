# CTT-JEPA: Causal Transformation Theory for Joint Embedding Predictive Architectures

> **Extending [C-JEPA](https://github.com/galilai-group/cjepa) with causal inductive biases from Causal Transformation Theory (CTT)**

## Key Idea

C-JEPA learns a causal world model through **masked slot prediction** — masking object slots and predicting them from context. We add two loss terms derived from our own [Causal Transformation Theory](ctt-builders-guide.md) that enforce the attention-based causal graph to satisfy fundamental causal axioms:

| Loss | CTT Axiom | What it enforces |
|---|---|---|
| **Sufficiency** | Axiom 4 | A slot's causal neighborhood alone should *suffice* to predict it |
| **Invariance** | Axiom 6 | Masking a non-interacting slot should *not* degrade predictions for unrelated slots |

These losses use the transformer's own attention weights as the causal adjacency graph — **no architecture changes** are required. The only modification is two additive loss terms controlled via config flags.

## Results

Evaluated on **CLEVRER VQA** using the ALOE framework. Both _Baseline_ and _CTT-JEPA_ models trained for 30 epochs (world model) + 100 epochs (ALOE VQA) on 1× H100 with `|M|=2` masked slots and `batch_size=2048`.

### ALOE VQA Accuracy (Epoch 79)

| Metric | Baseline | CTT-JEPA | Δ | Paper (\|M\|=2) |
|---|---|---|---|---|
| **Descriptive** | **91.4%** | 89.8% | −1.6% | 91.0% |
| **Counterfactual** | **57.5%** | 49.4% | −8.1% | 50.3% |
| **Explanatory** | **84.9%** | 81.0% | −3.9% | 82.5% |
| **Predictive** | **80.1%** | 77.9% | −2.2% | 79.6% |
| **Multiple-choice** | **72.1%** | 66.7% | −5.4% | — |

> Our baseline matches or exceeds the paper's C-JEPA (V) results at |M|=2, validating the pipeline.
>
> CTT losses (invariance + sufficiency) via attention-based causal graphs degrade VQA by 2–8%.

### Analysis

The attention weights used as the causal adjacency matrix capture **statistical correlations, not causal structure**. This causes the invariance and sufficiency losses to enforce the wrong constraints — particularly harming counterfactual reasoning (−8.1%). Key takeaways:

- **Naive CTT integration is insufficient** — a proper causal discovery mechanism is needed
- **A learned causal graph** (e.g., DAG-GNN, NOTEARS) would likely improve results
- **Loss weights** (`inv=0.2, suf=0.1`) may need to be much smaller as soft regularizers

For full analysis, see [RESULTS.md](RESULTS.md).

## What Changed from C-JEPA

Only **3 files** were modified/added:

```
src/ctt_losses.py                              [NEW]  — CTT loss functions
src/train/train_causalwm_from_clevrer_slot.py  [MOD]  — Integrated CTT into compute_loss()
configs/config_train_causal_clevrer_slot.yaml  [MOD]  — Added CTT config flags
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

### Setup

```bash
git clone -b ctt-jepa https://github.com/GerardCB/cjepa.git
cd cjepa
bash scripts/ctt/setup_runpod.sh
```

> Tested on `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`

### Train world models

```bash
bash scripts/ctt/train_worldmodels.sh
```

Trains baseline C-JEPA (30 epochs) + CTT-JEPA (30 epochs, CTT from epoch 10), with slot rollouts.

### Evaluate with ALOE VQA

```bash
bash scripts/ctt/train_aloe.sh
```

Trains ALOE VQA on both rollouts. Check results:
```bash
grep 'Eval epoch' logs/aloe_baseline.log
grep 'Eval epoch' logs/aloe_ctt.log
```

**Total time: ~7.5 hours on 1× H100.**

### Manual run

```bash
export PYTHONPATH=$(pwd)

# Train CTT-JEPA
python src/train/train_causalwm_from_clevrer_slot.py \
    embedding_dir=./data/clevrer_videosaur_slots.pkl \
    batch_size=2048 trainer.max_epochs=30 \
    ctt_inv_weight=0.2 ctt_suf_weight=0.1 ctt_start_epoch=10 \
    rollout.save_rollout=true output_model_name=ctt_full
```

## Repository Structure

```
cjepa/
├── src/
│   ├── ctt_losses.py                    ← Our CTT loss functions
│   ├── aloe_train.py                    ← ALOE VQA training (patched)
│   ├── cjepa_predictor.py               ← MaskedSlotPredictor (unmodified)
│   ├── train/
│   │   └── train_causalwm_from_clevrer_slot.py  ← Modified training loop
│   └── third_party/                     ← Dependencies
├── configs/
│   └── config_train_causal_clevrer_slot.yaml    ← Modified config
├── scripts/ctt/
│   ├── setup_runpod.sh                  ← One-command environment setup
│   ├── train_worldmodels.sh             ← Train baseline + CTT world models
│   ├── train_aloe.sh                    ← Train ALOE VQA on both rollouts
│   └── fix_torchcodec.py               ← Patch torchcodec for CUDA compat
├── results/
│   └── logs/                            ← Training logs with eval metrics
├── RESULTS.md                           ← Full experiment analysis
└── data/                                ← Slot embeddings (downloaded)
```

## Acknowledgments

Built on top of [C-JEPA](https://github.com/galilai-group/cjepa) by the Galilai Group. CTT axioms adapted from Causal Transformation Theory.
