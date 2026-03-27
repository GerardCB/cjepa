"""
CTT Losses for C-JEPA: Invariance and Sufficiency from Causal Transformation Theory.

These losses are additive terms on top of the existing C-JEPA loss. They use the
transformer's own attention weights as the causal adjacency graph, penalizing:

  - Invariance (Axiom 6): masking slot i should not degrade predictions for slots
    j that don't interact with i (non-neighbors).
  - Sufficiency (Axiom 4): the neighborhood of a target slot should alone suffice
    to predict it accurately.

Usage:
    Inject into C-JEPA's `compute_loss()` by calling `ctt_invariance_loss()` and/or
    `ctt_sufficiency_loss()` with the predictor, history, target, and config.

Reference:
    CTT axioms as applied to object-centric world models. The attention-based
    adjacency is threshold-based: A[i,j] > threshold means j is a neighbor of i.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np


def _get_slot_attention(predictor, history, layer_idx=-1):
    """
    Run a forward pass through the predictor with attention extraction.

    Unlike predictor.attention_probing() (which is @torch.no_grad), this
    function allows gradients to flow for CTT loss backprop.

    Args:
        predictor: MaskedSlotPredictor instance
        history: (B, T_hist, S, D) - input history slots
        layer_idx: which transformer layer's attention to extract

    Returns:
        pred_embedding: (B, T_total, S, D) - predicted slots
        mask_indices: indices of masked slots
        slot_attn: (B, S, S) - slot-to-slot attention at the future timestep,
                   averaged over time. This serves as the causal adjacency.
    """
    B, T_hist, S, D = history.shape
    T_total = predictor.total_frames

    # Prepare input (same as forward(), but we need attention)
    x_input, masked_indices = predictor.prepare_input(history)

    # Flatten for transformer
    x_flat = rearrange(x_input, 'b t s d -> b (t s) d')

    # Forward with attention extraction
    out_flat, attn_weights_list = predictor.transformer(x_flat, return_attention=True)

    # Unflatten
    out = rearrange(out_flat, 'b (t s) d -> b t s d', t=T_total, s=S)
    out = predictor.to_out(out)

    # Extract slot-to-slot attention from the specified layer
    # attn_weights shape: (B, T*S, T*S)
    attn_weights = attn_weights_list[layer_idx]

    # Reshape to (B, T_total, S, T_total, S)
    attn_reshaped = rearrange(
        attn_weights, 'b (tq sq) (tk sk) -> b tq sq tk sk',
        tq=T_total, sq=S, tk=T_total, sk=S
    )

    # Extract slot-to-slot attention at the future prediction timestep
    # For the first future frame (t=T_hist), get how each slot attends to others
    # Average over all key timesteps to get a single (B, S, S) slot adjacency
    future_attn = attn_reshaped[:, T_hist, :, :, :]  # (B, S, T_total, S)
    slot_attn = future_attn.mean(dim=2)  # (B, S, S) — average over key timesteps

    return out, masked_indices, slot_attn


def ctt_invariance_loss(predictor, history, target, original_mask_indices, cfg):
    """
    Invariance Loss (CTT Axiom 6).

    Principle: Masking slot i should not degrade predictions for slot j if j
    is not influenced by i (j ∉ N(i) in the causal graph).

    Implementation:
    1. Run a clean forward pass to get attention weights (causal graph proxy).
    2. For a randomly chosen slot i, identify its non-neighbors from attention.
    3. Run a second forward pass with slot i additionally masked.
    4. Penalize prediction differences for non-neighbor slots.

    Args:
        predictor: MaskedSlotPredictor
        history: (B, T_hist, S, D)
        target: (B, T_pred, S, D)
        original_mask_indices: mask indices from the standard forward pass
        cfg: Hydra config with ctt_adj_threshold

    Returns:
        loss: scalar tensor
    """
    B, T_hist, S, D = history.shape
    device = history.device
    threshold = cfg.get("ctt_adj_threshold", 0.15)

    # Step 1: Get attention from clean forward (with gradients)
    pred_clean, _, slot_attn = _get_slot_attention(predictor, history)

    # Extract future predictions
    T_hist_cfg = cfg.dinowm.history_size
    pred_future_clean = pred_clean[:, T_hist_cfg:T_hist_cfg + cfg.dinowm.num_preds, :, :]

    # Step 2: Choose a random slot to additionally mask for invariance testing
    # Avoid slots that are already masked
    original_set = set(original_mask_indices.tolist()) if len(original_mask_indices) > 0 else set()
    available = [s for s in range(S) if s not in original_set]

    if len(available) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    target_slot_i = available[np.random.randint(len(available))]

    # Step 3: Get non-neighbors of slot i
    # slot_attn[:, :, i] = how much each slot attends to slot i (influence of i on others)
    influence_of_i = slot_attn[:, :, target_slot_i].mean(dim=0)  # (S,) — averaged over batch
    non_neighbors = [
        j for j in range(S)
        if j != target_slot_i
        and j not in original_set
        and influence_of_i[j].item() < threshold
    ]

    if len(non_neighbors) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Step 4: Create a modified forward pass with slot i additionally masked
    new_mask_list = list(original_set) + [target_slot_i]
    is_slot_masked = torch.zeros(S, dtype=torch.bool, device=device)
    for idx in new_mask_list:
        is_slot_masked[idx] = True
    extended_mask_indices = torch.tensor(new_mask_list, dtype=torch.long, device=device)

    # Manually construct masked input (bypassing the fixed-seed get_mask_indices)
    x_input_masked = _prepare_input_with_mask(predictor, history, is_slot_masked, extended_mask_indices)

    # Forward through transformer
    x_flat = rearrange(x_input_masked, 'b t s d -> b (t s) d')
    out_flat = predictor.transformer(x_flat)
    out = rearrange(out_flat, 'b (t s) d -> b t s d', t=predictor.total_frames, s=S)
    out = predictor.to_out(out)

    pred_future_masked = out[:, T_hist_cfg:T_hist_cfg + cfg.dinowm.num_preds, :, :]

    # Step 5: For non-neighbor slots, the prediction should be unchanged
    loss = 0.0
    for j in non_neighbors:
        loss += F.mse_loss(
            pred_future_masked[:, :, j, :],
            pred_future_clean[:, :, j, :].detach()
        )

    return loss / len(non_neighbors)


def ctt_sufficiency_loss(predictor, history, target, cfg):
    """
    Sufficiency Loss (CTT Axiom 4).

    Principle: The causal neighborhood of slot i should be sufficient to
    predict slot i. Masking non-neighbors should not hurt prediction of i.

    Implementation:
    1. Run a clean forward pass to identify the neighborhood of a target slot.
    2. Mask all non-neighbors.
    3. Predict the target slot from neighbors only.
    4. The prediction should still be accurate.

    Args:
        predictor: MaskedSlotPredictor
        history: (B, T_hist, S, D)
        target: (B, T_pred, S, D)
        cfg: Hydra config with ctt_adj_threshold

    Returns:
        loss: scalar tensor
    """
    B, T_hist, S, D = history.shape
    device = history.device
    threshold = cfg.get("ctt_adj_threshold", 0.15)

    # Step 1: Get attention (causal graph) — no grad needed for graph construction
    with torch.no_grad():
        _, _, slot_attn = _get_slot_attention(predictor, history)

    # Step 2: Pick a random target slot
    target_slot = np.random.randint(S)

    # Step 3: Find non-neighbors of target slot
    # slot_attn[:, target_slot, :] = what target_slot attends to (its input neighbors)
    neighbors_of_target = slot_attn[:, target_slot, :].mean(dim=0)  # (S,)
    non_neighbors = [
        j for j in range(S)
        if j != target_slot and neighbors_of_target[j].item() < threshold
    ]

    if len(non_neighbors) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Step 4: Mask non-neighbors, keep target_slot and its neighbors visible
    is_slot_masked = torch.zeros(S, dtype=torch.bool, device=device)
    for idx in non_neighbors:
        is_slot_masked[idx] = True
    mask_indices = torch.tensor(non_neighbors, dtype=torch.long, device=device)

    x_input_masked = _prepare_input_with_mask(predictor, history, is_slot_masked, mask_indices)

    # Forward pass (with gradients)
    x_flat = rearrange(x_input_masked, 'b t s d -> b (t s) d')
    out_flat = predictor.transformer(x_flat)
    out = rearrange(out_flat, 'b (t s) d -> b t s d', t=predictor.total_frames, s=S)
    out = predictor.to_out(out)

    T_hist_cfg = cfg.dinowm.history_size
    pred_future = out[:, T_hist_cfg:T_hist_cfg + cfg.dinowm.num_preds, :, :]

    # Step 5: Prediction of target slot should still match ground truth
    loss = F.mse_loss(pred_future[:, :, target_slot, :], target[:, :, target_slot, :].detach())

    return loss


def _prepare_input_with_mask(predictor, x, is_slot_masked, masked_indices):
    """
    Prepare transformer input with a custom slot mask, bypassing
    predictor.get_mask_indices() which uses a fixed seed.

    This reproduces MaskedSlotPredictor.prepare_input() logic but with
    arbitrary mask indices.

    Args:
        predictor: MaskedSlotPredictor
        x: (B, T_hist, S, D)
        is_slot_masked: (S,) bool tensor — True for masked slots
        masked_indices: (M,) long tensor — indices of masked slots

    Returns:
        final_input: (B, T_total, S, D)
    """
    B, T_hist, S, D = x.shape
    T_total = predictor.total_frames
    device = x.device

    # Anchors
    anchors = x[:, 0, :, :]
    anchor_queries = predictor.id_projector(anchors)

    # Query grid
    tokens_grid = predictor.mask_token.expand(B, T_total, S, D)
    pos_grid = predictor.time_pos_embed.expand(B, T_total, S, D)
    anchor_grid = anchor_queries.unsqueeze(1).expand(B, T_total, S, D)
    query_input = tokens_grid + pos_grid + anchor_grid

    final_input = query_input.clone()

    # t=0: always visible for all slots
    final_input[:, 0, :, :] = x[:, 0, :, :] + predictor.time_pos_embed[:, 0, :, :]

    # Unmasked slots: visible at t=1..T_hist-1
    unmasked_indices = torch.where(~is_slot_masked)[0]

    if len(unmasked_indices) > 0 and T_hist > 1:
        real_history = x[:, 1:, unmasked_indices, :]
        history_pos = predictor.time_pos_embed[:, 1:T_hist, :, :].expand(B, T_hist - 1, S, D)
        history_pos_unmasked = history_pos[:, :, unmasked_indices, :]
        final_input[:, 1:T_hist, unmasked_indices, :] = real_history + history_pos_unmasked

    return final_input
