"""
model_highlight.py — Highlight Prediction Models
=================================================

Two model families, matching highlight_task.md design:

1. CausalHighlightTransformer  (KuaiHL-style)
   Causal multi-head self-attention — each segment can only attend to past
   segments. Position t predicts the highlight score of segment t+1.

2. HierarchicalHighlightTransformer  (AntPivot-style causal adaptation)
   Causal local-window attention followed by causal global attention. This
   preserves the hierarchical structure without exposing the target segment.

3. Baseline models (GRU, MLP)
   Used as comparison baselines.

4. HighlightLoss
   Mixed loss = point MSE + pairwise BPR ranking + border-aware MSE (KuaiHL).

All models accept batched input from collate_fn:
    embeddings : [B, T, 128]
    pad_mask   : [B, T]   bool, True = padded position (to ignore)
    stats      : [B, T, stat_dim] or None — optional leakage-free segment
                 statistics; consumed only when the model is built with
                 use_stats=True (additive branch, gate-init=0 → off == original).
and return:
    scores     : [B, T]   predicted highlight score in [0, 1]
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────────────────────

class _ScoreHead(nn.Module):
    """Linear → ReLU → Linear → Sigmoid, mapping d_model → 1."""
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d_model]  →  [...] (squeeze last dim)
        return self.fc(x).squeeze(-1)


class LearnedPositionalEncoding(nn.Module):
    """Learnable positional embedding table (max_len positions)."""
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)   # [1, T]
        return x + self.pe(pos)                                # [B, T, D]


class _StatBranch(nn.Module):
    """Optional additive correction from segment statistics features.

    When ``use_stats`` is on, the parent model adds ``_StatBranch(stats)`` to
    its embedding projection. The scalar ``gate`` is initialised to **0**, so
    at the start of training the stat contribution is exactly zero and the
    forward is bit-for-bit identical to the no-stat model; the gate then learns
    to open up as stat signal proves useful. ``gate == 0`` → bit-exact
    equivalence with the original (embedding-only) model.
    """
    def __init__(self, stat_dim: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(stat_dim, d_model)
        self.gate = nn.Parameter(torch.zeros(1))   # scalar, starts at 0

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        # stats: [B, T, stat_dim] → [B, T, d_model]
        return self.gate * self.proj(stats)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Causal Transformer  (KuaiHL-style, Online)
# ──────────────────────────────────────────────────────────────────────────────

class CausalHighlightTransformer(nn.Module):
    """
    Causal (left-to-right) Transformer for highlight score prediction.

    Architecture
    ------------
    Segment emb (128) → Linear proj → Learned PE → N × CausalTransformerLayer
    → Score head → sigmoid → [B, T]

    Each position attends only to itself and earlier positions (upper-triangle
    masked), so prediction at step k uses only segments 1 … k.

    Parameters
    ----------
    seg_dim    : input embedding dimension (128 for live_emb_128_ts)
    d_model    : Transformer hidden size
    n_heads    : number of attention heads  (d_model % n_heads == 0)
    n_layers   : number of Transformer encoder layers
    d_ff       : feed-forward hidden size  (default 4 * d_model)
    dropout    : dropout rate
    max_seq_len: maximum sequence length for positional encoding
    """

    def __init__(
        self,
        seg_dim:     int = 128,
        d_model:     int = 256,
        n_heads:     int = 4,
        n_layers:    int = 2,
        d_ff:        Optional[int] = None,
        dropout:     float = 0.1,
        max_seq_len: int = 512,
        use_stats:   bool = False,
        stat_dim:    int = 10,
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.proj = nn.Linear(seg_dim, d_model)
        self.pe   = LearnedPositionalEncoding(d_model, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                                  enable_nested_tensor=False)
        self.score_head  = _ScoreHead(d_model)

        # Optional additive stat-feature branch (gate-init=0 → off == original).
        self.stat_branch = _StatBranch(stat_dim, d_model) if use_stats else None

        self._d_model = d_model

    def forward(
        self,
        embeddings: torch.Tensor,   # [B, T, 128]
        pad_mask:   torch.Tensor,   # [B, T] bool, True = pad
        stats:      Optional[torch.Tensor] = None,   # [B, T, stat_dim] or None
    ) -> torch.Tensor:
        """
        Returns
        -------
        scores : [B, T]  predicted highlight scores (0–1), padded positions are 0.
        """
        B, T, _ = embeddings.shape
        x = self.pe(self.proj(embeddings))   # [B, T, d_model]
        if self.stat_branch is not None:
            if stats is None:
                raise ValueError("use_stats=True model requires a stats tensor")
            x = x + self.stat_branch(stats)  # [B, T, d_model]

        # Causal (autoregressive) mask: position i cannot see position j > i
        causal_mask = torch.triu(
            torch.ones(T, T, device=embeddings.device, dtype=torch.bool),
            diagonal=1,
        )   # [T, T]

        h = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=pad_mask,
            is_causal=True,
        )   # [B, T, d_model]

        scores = self.score_head(h)          # [B, T]
        # Zero-out padded positions
        scores = scores.masked_fill(pad_mask, 0.0)
        return scores


# ──────────────────────────────────────────────────────────────────────────────
# 2. Hierarchical Transformer  (AntPivot-style causal adaptation)
# ──────────────────────────────────────────────────────────────────────────────

class HierarchicalHighlightTransformer(nn.Module):
    """
    Hierarchical causal attention for one-step-ahead segment prediction.

    Architecture
    ------------
    Segment emb (128) → Linear proj → Learned PE
    → Local causal attention (window W, 1 layer)
    → Global causal attention (N_global layers)
    → Score head → sigmoid → [B, T]

    At position t, both stages can access only positions <= t. The output at t
    predicts segment t+1 without seeing its embedding.

    Parameters
    ----------
    seg_dim    : input embedding dimension (128)
    d_model    : hidden size
    n_heads    : attention heads
    n_global   : number of global attention layers
    window     : local causal window (current plus window-1 past segments)
    d_ff       : feed-forward size (default 4 * d_model)
    dropout    : dropout rate
    max_seq_len: max positional encoding length
    """

    def __init__(
        self,
        seg_dim:     int = 128,
        d_model:     int = 256,
        n_heads:     int = 4,
        n_global:    int = 2,
        window:      int = 5,
        d_ff:        Optional[int] = None,
        dropout:     float = 0.1,
        max_seq_len: int = 512,
        use_stats:   bool = False,
        stat_dim:    int = 10,
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.window = window

        self.proj = nn.Linear(seg_dim, d_model)
        self.pe   = LearnedPositionalEncoding(d_model, max_seq_len)

        # Local layer: 1 Transformer encoder layer (we apply a local mask)
        local_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
        )
        self.local_attn = nn.TransformerEncoder(local_layer, num_layers=1,
                                                 enable_nested_tensor=False)

        # Global layers receive a causal mask in forward().
        global_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True,
        )
        self.global_attn = nn.TransformerEncoder(global_layer, num_layers=n_global,
                                                  enable_nested_tensor=False)

        self.score_head = _ScoreHead(d_model)

        # Optional additive stat-feature branch (gate-init=0 → off == original).
        self.stat_branch = _StatBranch(stat_dim, d_model) if use_stats else None

    @staticmethod
    def _local_mask(T: int, window: int, device: torch.device) -> torch.Tensor:
        """
        Position i may attend only to itself and the preceding window-1
        positions. True entries are blocked.
        """
        idx = torch.arange(T, device=device)
        lag = idx.unsqueeze(1) - idx.unsqueeze(0)
        allowed = (lag >= 0) & (lag < window)
        return ~allowed

    def forward(
        self,
        embeddings: torch.Tensor,   # [B, T, 128]
        pad_mask:   torch.Tensor,   # [B, T] bool
        stats:      Optional[torch.Tensor] = None,   # [B, T, stat_dim] or None
    ) -> torch.Tensor:
        """
        Returns
        -------
        scores : [B, T]  (padded positions = 0)
        """
        B, T, _ = embeddings.shape
        x = self.pe(self.proj(embeddings))   # [B, T, d_model]
        if self.stat_branch is not None:
            if stats is None:
                raise ValueError("use_stats=True model requires a stats tensor")
            x = x + self.stat_branch(stats)  # [B, T, d_model]

        # Local attention with windowed mask
        local_mask = self._local_mask(T, self.window, embeddings.device)  # [T, T]
        h_local = self.local_attn(x, mask=local_mask, src_key_padding_mask=pad_mask)

        # Position t must not see the embedding at t+1, which is the target.
        causal_mask = torch.triu(
            torch.ones(T, T, device=embeddings.device, dtype=torch.bool),
            diagonal=1,
        )
        h_global = self.global_attn(
            h_local,
            mask=causal_mask,
            src_key_padding_mask=pad_mask,
            is_causal=True,
        )

        scores = self.score_head(h_global)         # [B, T]
        scores = scores.masked_fill(pad_mask, 0.0)
        return scores


# ──────────────────────────────────────────────────────────────────────────────
# 3. Baseline models
# ──────────────────────────────────────────────────────────────────────────────

class MLPHighlight(nn.Module):
    """Use segment t alone to predict t+1; no sequence modelling."""
    def __init__(self, seg_dim: int = 128, hidden: int = 256, dropout: float = 0.1,
                 use_stats: bool = False, stat_dim: int = 10) -> None:
        super().__init__()
        # First layer pulled out as a named proj so the stat branch can be added
        # to its output. When use_stats=False this is bit-equivalent to the
        # original inline nn.Linear(seg_dim, hidden).
        self.proj = nn.Linear(seg_dim, hidden)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )
        # Optional additive stat-feature branch (gate-init=0 → off == original).
        self.stat_branch = _StatBranch(stat_dim, hidden) if use_stats else None

    def forward(self, embeddings: torch.Tensor, pad_mask: torch.Tensor,
                stats: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.proj(embeddings)                # [B, T, hidden]
        if self.stat_branch is not None:
            if stats is None:
                raise ValueError("use_stats=True model requires a stats tensor")
            x = x + self.stat_branch(stats)      # [B, T, hidden]
        scores = self.net(x).squeeze(-1)         # [B, T]
        return scores.masked_fill(pad_mask, 0.0)


class GRUHighlight(nn.Module):
    """
    Causal GRU: processes segments left-to-right and predicts t+1 at step t.
    Same "online" property as CausalHighlightTransformer but with GRU.
    """
    def __init__(
        self,
        seg_dim:    int = 128,
        hidden:     int = 256,
        n_layers:   int = 2,
        dropout:    float = 0.1,
        use_stats:  bool = False,
        stat_dim:   int = 10,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(seg_dim, hidden)
        self.gru  = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.score_head = _ScoreHead(hidden)

        # Optional additive stat-feature branch (gate-init=0 → off == original).
        self.stat_branch = _StatBranch(stat_dim, hidden) if use_stats else None

    def forward(self, embeddings: torch.Tensor, pad_mask: torch.Tensor,
                stats: Optional[torch.Tensor] = None) -> torch.Tensor:
        x      = self.proj(embeddings)           # [B, T, hidden]
        if self.stat_branch is not None:
            if stats is None:
                raise ValueError("use_stats=True model requires a stats tensor")
            x = x + self.stat_branch(stats)      # [B, T, hidden]
        h, _   = self.gru(x)                     # [B, T, hidden]
        scores = self.score_head(h)              # [B, T]
        return scores.masked_fill(pad_mask, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# 4. HighlightLoss  (KuaiHL-style mixed loss)
# ──────────────────────────────────────────────────────────────────────────────

class HighlightLoss(nn.Module):
    """
    Mixed loss = λ_point * L_point  +  λ_pair * L_pair  +  λ_border * L_border

    L_point  : MSE between predicted and target highlight scores
    L_pair   : Pairwise BPR ranking loss over segment pairs with |Δscore| > delta
    L_border : Border-aware MSE: higher weight at segment boundaries (sharp
               transitions in true score), following KuaiHL's Border-Aware loss

    Parameters
    ----------
    lambda_point  : weight for point MSE loss
    lambda_pair   : weight for pairwise ranking loss
    lambda_border : weight for border-aware MSE
    pair_delta    : minimum true-score difference to form a training pair
    border_delta  : threshold for "significant" score change at a boundary
    border_gamma  : extra weight amplifier at boundaries
    """

    def __init__(
        self,
        lambda_point:  float = 1.0,
        lambda_pair:   float = 0.5,
        lambda_border: float = 0.3,
        pair_delta:    float = 0.2,
        border_delta:  float = 0.15,
        border_gamma:  float = 2.0,
    ) -> None:
        super().__init__()
        self.lambda_point  = lambda_point
        self.lambda_pair   = lambda_pair
        self.lambda_border = lambda_border
        self.pair_delta    = pair_delta
        self.border_delta  = border_delta
        self.border_gamma  = border_gamma

    def forward(
        self,
        pred:     torch.Tensor,   # [B, T]  predicted scores
        target:   torch.Tensor,   # [B, T]  true hl_scores  (-1 for padded)
        pad_mask: torch.Tensor,   # [B, T]  bool, True = padded
    ) -> torch.Tensor:
        """
        Returns scalar loss.
        Padded positions (pad_mask=True) are excluded from all terms.
        """
        valid = ~pad_mask   # [B, T] bool, True = real position

        # ── Point loss ────────────────────────────────────────────────────────
        # MSE only on valid positions
        err   = (pred - target) ** 2          # [B, T]
        L_point = err[valid].mean() if valid.any() else torch.tensor(0.0, device=pred.device)

        # ── Pairwise ranking loss ──────────────────────────────────────────────
        # For each sequence, consider all valid pairs (i, j) with target_i - target_j > delta.
        L_pair_list = []
        B, T = pred.shape
        for b in range(B):
            v = valid[b]                             # [T] bool
            if v.sum() < 2:
                continue
            p_b = pred[b][v]                         # [K]
            t_b = target[b][v]                       # [K]

            diff_t  = t_b.unsqueeze(0) - t_b.unsqueeze(1)   # [K, K] t_i - t_j
            diff_p  = p_b.unsqueeze(0) - p_b.unsqueeze(1)   # [K, K] p_i - p_j
            pos_pair = diff_t > self.pair_delta               # i should rank higher than j

            if pos_pair.any():
                bpr = -F.logsigmoid(diff_p[pos_pair])
                L_pair_list.append(bpr.mean())

        if L_pair_list:
            L_pair = torch.stack(L_pair_list).mean()
        else:
            L_pair = torch.tensor(0.0, device=pred.device)

        # ── Border-aware MSE ──────────────────────────────────────────────────
        # w_k is amplified when the score transitions sharply at position k or k-1
        # Use target score difference between adjacent valid positions.
        weights = torch.ones_like(target)   # [B, T]
        if T > 1:
            # score difference between adjacent positions (left shift)
            delta_score = (target[:, 1:] - target[:, :-1]).abs()    # [B, T-1]
            is_border   = delta_score > self.border_delta             # [B, T-1] bool
            amp         = self.border_gamma * is_border.float()
            # amplify position k-1 and k when transition is at [k-1 → k]
            weights[:, :-1] = weights[:, :-1] + amp
            weights[:, 1:]  = weights[:, 1:]  + amp
        # zero weight for padded positions
        weights = weights * valid.float()
        n_valid = weights.sum().clamp(min=1.0)
        L_border = (weights * err).sum() / n_valid

        total = (
            self.lambda_point  * L_point
            + self.lambda_pair   * L_pair
            + self.lambda_border * L_border
        )
        return total


# ──────────────────────────────────────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "causal":        CausalHighlightTransformer,
    "hierarchical":  HierarchicalHighlightTransformer,
    "gru":           GRUHighlight,
    "mlp":           MLPHighlight,
}


def build_model(name: str, **kwargs) -> nn.Module:
    """
    Build a model by name.

    Parameters
    ----------
    name : one of 'causal', 'hierarchical', 'gru', 'mlp'
    **kwargs: constructor arguments (passed verbatim)
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
