"""Sequential baselines adapted to the KLM3-SA QRec batch interface.

The implementations retain the core architecture of the public reference
repositories listed in the root ``README.md`` while sharing KLM3's event
splits, history construction, negative sampling, and evaluator.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _last_nonpad(hidden: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    # KLM3 histories are left padded, so sequence length - 1 is not the last
    # valid index for short histories. Find the rightmost non-padding token.
    positions = torch.arange(ids.size(1), device=ids.device).unsqueeze(0).expand_as(ids)
    last = positions.masked_fill(ids.eq(0), -1).max(dim=1).values.clamp_min(0)
    return hidden[torch.arange(hidden.size(0), device=ids.device), last]


def _right_padded(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert left-padded histories to right-padded form for recurrent models."""
    length = ids.ne(0).sum(dim=1)
    width = ids.size(1)
    starts = (width - length).unsqueeze(1)
    source = torch.arange(width, device=ids.device).unsqueeze(0) + starts
    source = source.clamp_max(width - 1)
    compact = ids.gather(1, source)
    compact = compact.masked_fill(torch.arange(width, device=ids.device).unsqueeze(0) >= length.unsqueeze(1), 0)
    return compact, length


class SequentialBaseline(nn.Module):
    def score(self, uid: torch.Tensor, rec_his: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, batch) -> torch.Tensor:
        pos = self.score(batch["uid"], batch["rec_his"], batch["pos"])
        neg = self.score(batch["uid"], batch["rec_his"], batch["neg"])
        return F.softplus(-pos).mean() + F.softplus(neg).mean()

    @staticmethod
    def _candidate_score(vector: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        return ((vector.unsqueeze(1) * item).sum(-1) if item.ndim == 3
                else (vector * item).sum(-1))


class Caser(SequentialBaseline):
    """Caser: vertical and horizontal convolutions over an item sequence."""
    def __init__(self, n_users: int, n_items: int, dim: int, max_len: int,
                 dropout: float = 0.2, n_h: int = 16, n_v: int = 4) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, dim)
        self.item_embedding = nn.Embedding(n_items, dim, padding_idx=0)
        self.vertical = nn.Conv2d(1, n_v, (max_len, 1))
        self.horizontal = nn.ModuleList([nn.Conv2d(1, n_h, (height, dim))
                                         for height in range(1, max_len + 1)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_v * dim + n_h * max_len, dim)
        self.output_embedding = nn.Embedding(n_items, dim * 2, padding_idx=0)
        self.output_bias = nn.Embedding(n_items, 1, padding_idx=0)

    def _vector(self, uid: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        x = self.item_embedding(history).unsqueeze(1)
        vertical = self.vertical(x).reshape(x.size(0), -1)
        horizontal = []
        for conv in self.horizontal:
            activation = F.relu(conv(x).squeeze(-1))
            horizontal.append(F.max_pool1d(activation, activation.size(-1)).squeeze(-1))
        features = torch.cat([vertical, *horizontal], dim=-1)
        z = F.relu(self.fc(self.dropout(features)))
        return torch.cat([z, self.user_embedding(uid)], dim=-1)

    def score(self, uid, rec_his, items):
        vector = self._vector(uid, rec_his)
        item = self.output_embedding(items)
        bias = self.output_bias(items).squeeze(-1)
        return self._candidate_score(vector, item) + bias


class HGN(SequentialBaseline):
    """HGN: feature gating plus instance gating over sequence items."""
    def __init__(self, n_users: int, n_items: int, dim: int, max_len: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, dim)
        self.item_embedding = nn.Embedding(n_items, dim, padding_idx=0)
        self.feature_item = nn.Linear(dim, dim)
        self.feature_user = nn.Linear(dim, dim)
        self.instance_item = nn.Parameter(torch.empty(dim, 1))
        self.instance_user = nn.Parameter(torch.empty(dim, max_len))
        self.output_embedding = nn.Embedding(n_items, dim, padding_idx=0)
        self.output_bias = nn.Embedding(n_items, 1, padding_idx=0)
        nn.init.xavier_uniform_(self.instance_item)
        nn.init.xavier_uniform_(self.instance_user)

    def _components(self, uid: torch.Tensor, history: torch.Tensor):
        user = self.user_embedding(uid)
        item = self.item_embedding(history)
        gate = torch.sigmoid(self.feature_item(item) + self.feature_user(user).unsqueeze(1))
        gated = item * gate
        score = torch.sigmoid(gated.matmul(self.instance_item).squeeze(-1) + user.matmul(self.instance_user))
        score = score.masked_fill(history.eq(0), 0.0)
        union = (gated * score.unsqueeze(-1)).sum(1) / score.sum(1, keepdim=True).clamp_min(1e-8)
        return user, item, union

    def score(self, uid, rec_his, items):
        user, history_item, union = self._components(uid, rec_his)
        output = self.output_embedding(items)
        bias = self.output_bias(items).squeeze(-1)
        if items.ndim == 1:
            relation = (history_item * output.unsqueeze(1)).sum(-1)
            relation = relation.masked_fill(rec_his.eq(0), 0.0).sum(-1)
            return (user * output).sum(-1) + (union * output).sum(-1) + relation + bias
        mf = (user.unsqueeze(1) * output).sum(-1)
        union_score = (union.unsqueeze(1) * output).sum(-1)
        relation = (history_item.unsqueeze(2) * output.unsqueeze(1)).sum(-1)
        relation = relation.masked_fill(rec_his.eq(0).unsqueeze(-1), 0.0).sum(1)
        return mf + union_score + relation + bias


class NARM(SequentialBaseline):
    """NARM: GRU global preference plus attention-based local preference."""
    def __init__(self, n_items: int, dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(n_items, dim, padding_idx=0)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.attn_encoder = nn.Linear(dim, dim, bias=False)
        self.attn_decoder = nn.Linear(dim, dim, bias=False)
        self.attn_score = nn.Linear(dim, 1, bias=False)
        self.project = nn.Linear(dim * 2, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _vector(self, history: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(self.dropout(self.item_embedding(history)))
        global_pref = _last_nonpad(output, history)
        score = self.attn_score(torch.sigmoid(self.attn_encoder(output) +
                                              self.attn_decoder(global_pref).unsqueeze(1))).squeeze(-1)
        score = score.masked_fill(history.eq(0), -1e9)
        weights = torch.softmax(score, dim=-1)
        weights = torch.where(history.ne(0), weights, torch.zeros_like(weights))
        local_pref = (output * weights.unsqueeze(-1)).sum(1)
        return self.project(torch.cat([global_pref, local_pref], dim=-1))

    def score(self, uid, rec_his, items):
        del uid
        return self._candidate_score(self._vector(rec_his), self.item_embedding(items))


class GRU4Rec(SequentialBaseline):
    """GRU4Rec-style recurrent session encoder with item-output scoring."""
    def __init__(self, n_items: int, dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(n_items, dim, padding_idx=0)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.output_embedding = nn.Embedding(n_items, dim, padding_idx=0)
        self.output_bias = nn.Embedding(n_items, 1, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def _vector(self, history: torch.Tensor) -> torch.Tensor:
        compact, length = _right_padded(history)
        output, _ = self.gru(self.dropout(self.item_embedding(compact)))
        last = (length.clamp_min(1) - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, output.size(-1))
        vector = output.gather(1, last).squeeze(1)
        return vector.masked_fill(length.eq(0).unsqueeze(1), 0.0)

    def score(self, uid, rec_his, items):
        del uid
        item = self.output_embedding(items)
        return self._candidate_score(self._vector(rec_his), item) + self.output_bias(items).squeeze(-1)


class FMLPBlock(nn.Module):
    """Frequency-domain filter followed by the FMLP-Rec feed-forward block."""
    def __init__(self, dim: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, max_len // 2 + 1, dim, 2) * 0.02)
        self.filter_dropout = nn.Dropout(dropout)
        self.filter_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, hidden: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(hidden, dim=1, norm="ortho")
        filtered = torch.fft.irfft(spectrum * torch.view_as_complex(self.complex_weight), n=hidden.size(1),
                                  dim=1, norm="ortho")
        hidden = self.filter_norm(hidden + self.filter_dropout(filtered))
        hidden = hidden.masked_fill(padding.unsqueeze(-1), 0.0)
        hidden = self.ffn_norm(hidden + self.ffn_dropout(self.ffn(hidden)))
        return hidden.masked_fill(padding.unsqueeze(-1), 0.0)


class FMLPRec(SequentialBaseline):
    """Filter-enhanced MLP sequential recommender (FMLP-Rec)."""
    def __init__(self, n_items: int, dim: int, max_len: int, dropout: float = 0.2,
                 num_blocks: int = 2) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(n_items, dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([FMLPBlock(dim, max_len, dropout) for _ in range(num_blocks)])

    def _vector(self, history: torch.Tensor) -> torch.Tensor:
        length = history.size(1)
        positions = torch.arange(length, device=history.device).unsqueeze(0)
        padding = history.eq(0)
        hidden = self.input_norm(self.item_embedding(history) + self.position_embedding(positions))
        hidden = self.dropout(hidden).masked_fill(padding.unsqueeze(-1), 0.0)
        for block in self.blocks:
            hidden = block(hidden, padding)
        return _last_nonpad(hidden, history)

    def score(self, uid, rec_his, items):
        del uid
        return self._candidate_score(self._vector(rec_his), self.item_embedding(items))


class SASRec(SequentialBaseline):
    """Causal self-attentive sequential recommendation (SASRec)."""
    def __init__(self, n_items: int, dim: int, max_len: int, dropout: float = 0.2,
                 num_heads: int = 2, num_blocks: int = 2) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("SASRec dim must be divisible by num_heads")
        self.item_embedding = nn.Embedding(n_items, dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, dim)
        # Keep to the PyTorch 1.10 API used by the original SAQRec environment.
        layer = nn.TransformerEncoderLayer(dim, num_heads, dim_feedforward=dim * 4,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_blocks)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def _vector(self, history: torch.Tensor) -> torch.Tensor:
        length = history.size(1)
        positions = torch.arange(length, device=history.device).unsqueeze(0)
        x = self.dropout(self.item_embedding(history) * math.sqrt(self.item_embedding.embedding_dim) +
                         self.position_embedding(positions))
        padding = history.eq(0)
        safe_padding = padding.clone()
        safe_padding[padding.all(dim=1), 0] = False
        causal = torch.triu(torch.ones(length, length, device=history.device, dtype=torch.bool), diagonal=1)
        hidden = self.encoder(x, mask=causal, src_key_padding_mask=safe_padding)
        hidden = hidden.masked_fill(padding.unsqueeze(-1), 0.0)
        return self.norm(_last_nonpad(hidden, history))

    def score(self, uid, rec_his, items):
        del uid
        return self._candidate_score(self._vector(rec_his), self.item_embedding(items))


BASELINE_MODELS = {
    "Caser": Caser,
    "FMLPRec": FMLPRec,
    "GRU4Rec": GRU4Rec,
    "HGN": HGN,
    "NARM": NARM,
    "SASRec": SASRec,
}


def build_baseline(name: str, n_users: int, n_items: int, dim: int, max_len: int,
                   dropout: float, num_heads: int, num_blocks: int) -> SequentialBaseline:
    try:
        cls = BASELINE_MODELS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown baseline {name}; choose one of {sorted(BASELINE_MODELS)}") from exc
    if name == "Caser":
        return cls(n_users, n_items, dim, max_len, dropout)
    if name == "HGN":
        return cls(n_users, n_items, dim, max_len)
    if name in {"NARM", "GRU4Rec"}:
        return cls(n_items, dim, dropout)
    if name == "FMLPRec":
        return cls(n_items, dim, max_len, dropout, num_blocks)
    return cls(n_items, dim, max_len, dropout, num_heads, num_blocks)
