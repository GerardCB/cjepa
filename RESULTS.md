# CTT-JEPA Benchmarking Results

## Overview

We integrated **Causal Temporal Theory (CTT)** invariance and sufficiency losses into the official **C-JEPA** world model and benchmarked on the **CLEVRER VQA** task against the unmodified baseline. The goal was to test whether CTT-derived causal constraints improve counterfactual and predictive reasoning.

> **Key Finding**: CTT losses (invariance + sufficiency) applied via attention-based causal graphs **degrade** VQA performance by 2-8% across all question types. The baseline remains stronger.

---

## Experimental Setup

| Parameter | Value |
|---|---|
| **GPU** | 1× NVIDIA H100 80GB (RunPod, on-demand) |
| **World Model** | C-JEPA with VideoSAUR slots |
| **Slot Encoder** | VideoSAUR (7 slots × 128 dim) |
| **Training Epochs** | 30 (world model), 100 (ALOE VQA) |
| **Batch Size** | 2048 |
| **Masked Slots \|M\|** | 2 |
| **ALOE Eval Interval** | Every 20 epochs |
| **Data** | CLEVRER (10K train, 5K val, 5K test videos) |
| **Total GPU Time** | ~7.5 hours |

### CTT Configuration

| Parameter | Value |
|---|---|
| `ctt_inv_weight` | 0.2 |
| `ctt_suf_weight` | 0.1 |
| `ctt_start_epoch` | 10 (phased, pure C-JEPA for first 10 epochs) |
| **Causal Graph Source** | Predictor's transformer attention weights |

---

## Results

### World Model Training (30 Epochs)

| Model | Final Train Loss | Final Val Loss |
|---|---|---|
| **Baseline C-JEPA** | 0.0150 | 0.0147 |
| **CTT-JEPA** | 0.0163 | 0.0150 |

### ALOE VQA Accuracy (Epoch 79)

| Metric | Baseline | CTT-JEPA | Δ | Paper (M=2) |
|---|---|---|---|---|
| **Descriptive** | **91.4%** | 89.8% | −1.6% | 91.0% |
| **Counterfactual** | **57.5%** | 49.4% | −8.1% | 50.3% |
| **Explanatory** | **84.9%** | 81.0% | −3.9% | 82.5% |
| **Predictive** | **80.1%** | 77.9% | −2.2% | 79.6% |
| **Multiple-choice** | **72.1%** | 66.7% | −5.4% | — |

> Our baseline **matches or exceeds** the paper's reported C-JEPA (V) results at |M|=2, validating the experimental pipeline.

### Accuracy Progression Over Training

| Metric | Ep 39 (Δ) | Ep 59 (Δ) | Ep 79 (Δ) |
|---|---|---|---|
| Descriptive | −1.7% | −1.3% | −1.6% |
| Counterfactual | −3.6% | −4.2% | **−8.1%** |
| Explanatory | −1.8% | −2.5% | −3.9% |
| Predictive | −1.0% | −2.9% | −2.2% |
| Multiple-choice | −2.5% | −3.3% | −5.4% |

---

## Analysis

### Why CTT Losses Hurt Performance

1. **Attention ≠ Causal Structure** — Transformer attention captures statistical correlations, not true causal relationships between slots. The invariance and sufficiency constraints were enforcing the wrong causal structure.

2. **Loss Weight Sensitivity** — The weights (`inv=0.2, suf=0.1`) may be too aggressive, overwhelming the primary dynamics prediction objective.

3. **Counterfactual Reasoning Hit Hardest** — The −8.1% drop on counterfactual questions suggests that enforcing invariance under wrong "interventions" directly harms counterfactual reasoning.

4. **Phased Start Insufficient** — Starting CTT at epoch 10 may not give the model enough time to learn stable dynamics. Attention patterns at epoch 10 are still noisy.

---

## Future Directions

1. **Learned Causal Graphs** — Replace attention-based adjacency with a differentiable graph learning module (e.g., NOTEARS, DAG-GNN)
2. **Lower Loss Weights** — Try `inv=0.01, suf=0.005` to make CTT a soft regularizer
3. **Later Phase Start** — Activate CTT at epoch 20+ for more stable attention patterns
4. **|M|=4 Training** — The paper's best results use 4 masked slots; this would provide a fairer comparison
5. **Separate Causal Module** — Train a small MLP to predict slot-to-slot causal relationships from the dynamics

---

## Reproducing

See [scripts/ctt/](scripts/ctt/) for the full RunPod pipeline:

```bash
# On RunPod H100
git clone -b ctt-jepa https://github.com/GerardCB/cjepa.git
cd cjepa
bash scripts/ctt/setup_runpod.sh

# Train both models (world model + rollout)
bash scripts/ctt/train_worldmodels.sh

# Run ALOE VQA evaluation
bash scripts/ctt/train_aloe.sh
```
