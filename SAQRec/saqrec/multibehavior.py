"""Questionnaire-aware multi-behavior baselines used in SAQRec Table 2."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .multibehavior_data import CLICK, DISSATISFIED, SATISFIED


def _last_nonpad(hidden: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(ids.size(1), device=ids.device).unsqueeze(0).expand_as(ids)
    last = positions.masked_fill(ids.eq(0), -1).max(dim=1).values.clamp_min(0)
    return hidden[torch.arange(hidden.size(0), device=ids.device), last]


def _safe_padding(mask: torch.Tensor) -> torch.Tensor:
    safe = mask.clone()
    safe[mask.all(dim=1), 0] = False
    return safe


def _right_pad_pair(ids: torch.Tensor, kinds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compact left-padded tokens before recurrent encoding."""
    length, width = ids.ne(0).sum(dim=1), ids.size(1)
    starts = (width - length).unsqueeze(1)
    source = (torch.arange(width, device=ids.device).unsqueeze(0) + starts).clamp_max(width - 1)
    compact_ids, compact_kinds = ids.gather(1, source), kinds.gather(1, source)
    padding = torch.arange(width, device=ids.device).unsqueeze(0) >= length.unsqueeze(1)
    return compact_ids.masked_fill(padding, 0), compact_kinds.masked_fill(padding, 0), length


class MixedBaseline(nn.Module):
    """Common ranking interface for all mixed-feedback baselines."""

    uses_mixed_history = True

    @staticmethod
    def _candidate_score(vector: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        return ((vector.unsqueeze(1) * item).sum(-1) if item.ndim == 3
                else (vector * item).sum(-1))

    def score(self, uid: torch.Tensor, mixed_his: torch.Tensor, mixed_type: torch.Tensor,
              items: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, batch) -> torch.Tensor:
        pos = self.score(batch["uid"], batch["mixed_his"], batch["mixed_type"], batch["pos"])
        neg = self.score(batch["uid"], batch["mixed_his"], batch["mixed_type"], batch["neg"])
        return F.softplus(-pos).mean() + F.softplus(neg).mean()


class MixedInput(nn.Module):
    def __init__(self, n_items: int, dim: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(n_items, dim, padding_idx=0)
        self.type_embedding = nn.Embedding(4, dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids: torch.Tensor, kinds: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        x = self.item_embedding(ids) + self.type_embedding(kinds) + self.position_embedding(positions)
        return self.dropout(self.norm(x)).masked_fill(ids.eq(0).unsqueeze(-1), 0.0)


class GRU4RecM(MixedBaseline):
    """GRU4Rec over the timestamp-mixed CLICK/SATISFIED/DISSATISFIED stream."""

    def __init__(self, n_items: int, dim: int, max_len: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.input = MixedInput(n_items, dim, max_len, dropout)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.output_embedding = nn.Embedding(n_items, dim, padding_idx=0)
        self.output_bias = nn.Embedding(n_items, 1, padding_idx=0)

    def _vector(self, ids: torch.Tensor, kinds: torch.Tensor) -> torch.Tensor:
        ids, kinds, length = _right_pad_pair(ids, kinds)
        output, _ = self.gru(self.input(ids, kinds))
        last = (length.clamp_min(1) - 1).view(-1, 1, 1).expand(-1, 1, output.size(-1))
        vector = output.gather(1, last).squeeze(1)
        return vector.masked_fill(length.eq(0).unsqueeze(1), 0.0)

    def score(self, uid, mixed_his, mixed_type, items):
        del uid
        return self._candidate_score(self._vector(mixed_his, mixed_type), self.output_embedding(items)) + \
            self.output_bias(items).squeeze(-1)


class SASRecM(MixedBaseline):
    """SASRec with a feedback-type embedding in its mixed behavior sequence."""

    def __init__(self, n_items: int, dim: int, max_len: int, dropout: float = 0.2,
                 num_heads: int = 2, num_blocks: int = 2) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("SASRecM dim must be divisible by num_heads")
        self.input = MixedInput(n_items, dim, max_len, dropout)
        layer = nn.TransformerEncoderLayer(dim, num_heads, dim_feedforward=dim * 4,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_blocks)
        self.norm = nn.LayerNorm(dim)

    def _vector(self, ids: torch.Tensor, kinds: torch.Tensor) -> torch.Tensor:
        length = ids.size(1)
        causal = torch.triu(torch.ones(length, length, device=ids.device, dtype=torch.bool), diagonal=1)
        padding = ids.eq(0)
        hidden = self.encoder(self.input(ids, kinds), mask=causal, src_key_padding_mask=_safe_padding(padding))
        hidden = hidden.masked_fill(padding.unsqueeze(-1), 0.0)
        return self.norm(_last_nonpad(hidden, ids))

    def score(self, uid, mixed_his, mixed_type, items):
        del uid
        return self._candidate_score(self._vector(mixed_his, mixed_type), self.input.item_embedding(items))


class FMLPBlock(nn.Module):
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


class FMLPRecM(MixedBaseline):
    """FMLP-Rec over the timestamp-mixed feedback stream."""

    def __init__(self, n_items: int, dim: int, max_len: int, dropout: float = 0.2,
                 num_blocks: int = 2) -> None:
        super().__init__()
        self.input = MixedInput(n_items, dim, max_len, dropout)
        self.blocks = nn.ModuleList([FMLPBlock(dim, max_len, dropout) for _ in range(num_blocks)])

    def _vector(self, ids: torch.Tensor, kinds: torch.Tensor) -> torch.Tensor:
        padding = ids.eq(0)
        hidden = self.input(ids, kinds)
        for block in self.blocks:
            hidden = block(hidden, padding)
        return _last_nonpad(hidden, ids)

    def score(self, uid, mixed_his, mixed_type, items):
        del uid
        return self._candidate_score(self._vector(mixed_his, mixed_type), self.input.item_embedding(items))


class FeedRec(MixedBaseline):
    """FeedRec adaptation for CLICK plus questionnaire S+/S- feedback.

    The original framework also includes skip, finish, quick-close and dwell
    feedback.  Those do not exist under this benchmark's interaction-only
    protocol, so this Table-2 version retains its heterogeneous/homogeneous
    behavior encoders and strong-to-weak interest distillation only.
    """

    def __init__(self, n_items: int, dim: int, max_len: int, dropout: float = 0.2,
                 num_heads: int = 2, num_blocks: int = 2, disentangle_weight: float = 0.1) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("FeedRec dim must be divisible by num_heads")
        self.input = MixedInput(n_items, dim, max_len, dropout)
        layer = nn.TransformerEncoderLayer(dim, num_heads, dim_feedforward=dim * 4,
                                           dropout=dropout, batch_first=True)
        self.heterogeneous = nn.TransformerEncoder(layer, num_layers=num_blocks)
        self.homogeneous = nn.ModuleDict({
            str(kind): nn.TransformerEncoder(
                nn.TransformerEncoderLayer(dim, num_heads, dim_feedforward=dim * 4,
                                           dropout=dropout, batch_first=True), num_layers=1
            ) for kind in (CLICK, SATISFIED, DISSATISFIED)
        })
        self.type_queries = nn.ParameterDict({str(kind): nn.Parameter(torch.empty(dim))
                                              for kind in (CLICK, SATISFIED, DISSATISFIED)})
        self.fuse = nn.Sequential(nn.Linear(dim * 5, dim), nn.ReLU(), nn.Linear(dim, 5))
        self.disentangle_weight = disentangle_weight
        for query in self.type_queries.values():
            nn.init.normal_(query, std=0.02)

    @staticmethod
    def _attention(values: torch.Tensor, mask: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        scores = (values * query.unsqueeze(1)).sum(-1).masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        return (values * weights.unsqueeze(-1)).sum(1)

    def _representations(self, ids: torch.Tensor, kinds: torch.Tensor):
        padding = ids.eq(0)
        hidden = self.heterogeneous(self.input(ids, kinds), src_key_padding_mask=_safe_padding(padding))
        hidden = hidden.masked_fill(padding.unsqueeze(-1), 0.0)
        vectors = {}
        for kind in (CLICK, SATISFIED, DISSATISFIED):
            keep = kinds.eq(kind)
            encoded = self.homogeneous[str(kind)](hidden.masked_fill(~keep.unsqueeze(-1), 0.0),
                                                   src_key_padding_mask=_safe_padding(~keep))
            encoded = encoded.masked_fill(~keep.unsqueeze(-1), 0.0)
            query = self.type_queries[str(kind)].unsqueeze(0).expand(ids.size(0), -1)
            vectors[kind] = self._attention(encoded, keep, query)
        click_mask = kinds.eq(CLICK)
        pos_from_click = self._attention(hidden, click_mask, vectors[SATISFIED])
        neg_from_click = self._attention(hidden, click_mask, vectors[DISSATISFIED])
        components = torch.stack([vectors[CLICK], vectors[SATISFIED], vectors[DISSATISFIED],
                                  pos_from_click, neg_from_click], dim=1)
        weights = torch.softmax(self.fuse(components.flatten(1)), dim=-1)
        vector = (components * weights.unsqueeze(-1)).sum(1)
        return vector, pos_from_click, neg_from_click

    def score(self, uid, mixed_his, mixed_type, items):
        del uid
        vector, _, _ = self._representations(mixed_his, mixed_type)
        return self._candidate_score(vector, self.input.item_embedding(items))

    def loss(self, batch) -> torch.Tensor:
        vector, pos_interest, neg_interest = self._representations(batch["mixed_his"], batch["mixed_type"])
        pos = self._candidate_score(vector, self.input.item_embedding(batch["pos"]))
        neg = self._candidate_score(vector, self.input.item_embedding(batch["neg"]))
        ranking = F.softplus(-pos).mean() + F.softplus(neg).mean()
        cosine = F.cosine_similarity(pos_interest, neg_interest, dim=-1, eps=1e-8)
        return ranking + self.disentangle_weight * cosine.square().mean()


MULTIBEHAVIOR_MODELS = {
    "FeedRec": FeedRec,
    "GRU4RecM": GRU4RecM,
    "SASRecM": SASRecM,
    "FMLPRecM": FMLPRecM,
}


class DINAttention(nn.Module):
    """DFN-style target attention over a masked memory bank.

    Implements ``[q, k, q*k, q-k] -> MLP -> softmax(masked) -> weighted sum``
    from the original DFN ``attention()`` op, evaluated against a chosen mask
    (typically the CLICK-token positions).
    """

    def __init__(self, dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim * 4, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, values: torch.Tensor, mask: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        query_expand = query.unsqueeze(1).expand_as(values)
        features = torch.cat([query_expand, values, query_expand * values, query_expand - values], dim=-1)
        scores = self.mlp(features).squeeze(-1) * self.scale
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        return (values * weights.unsqueeze(-1)).sum(1)


class DFN(MixedBaseline):
    """Deep Feedback Network adapted to questionnaire behaviors.

    Semantic mapping (paper role -> KLM3 token):
        clicked_seq  <- SATISFIED   (strong positive)
        unclick_seq  <- CLICK       (abundant, shared cross-attention memory)
        feedback_seq <- DISSATISFIED (strong negative)

    Downstream is simplified from DFN's FM+Deep+Wide+sigmoid to a 6-way MLP
    fusion that emits a single user vector scored against candidates via inner
    product, matching FeedRec/SASRecM's softplus-BPR retrieval protocol.
    """

    def __init__(self, n_users: int, n_items: int, dim: int, max_len: int, dropout: float = 0.2,
                 num_heads: int = 2, num_blocks: int = 2) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("DFN dim must be divisible by num_heads")
        self.input = MixedInput(n_items, dim, max_len, dropout)
        self.user_embedding = nn.Embedding(n_users, dim, padding_idx=0)
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        make_encoder = lambda: nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, num_heads, dim_feedforward=dim * 4,
                                       dropout=dropout, batch_first=True),
            num_layers=num_blocks,
        )
        self.enc_click = make_encoder()
        self.enc_middle = make_encoder()
        self.enc_feedback = make_encoder()
        self.attn_click = DINAttention(dim)
        self.attn_feedback = DINAttention(dim)
        self.fuse = nn.Sequential(
            nn.Linear(dim * 6, dim), nn.GELU(), nn.Dropout(dropout),
            nn.LayerNorm(dim), nn.Linear(dim, dim),
        )

    def _encode(self, encoder: nn.TransformerEncoder, hidden: torch.Tensor,
                keep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = encoder(hidden.masked_fill(~keep.unsqueeze(-1), 0.0),
                          src_key_padding_mask=_safe_padding(~keep))
        encoded = encoded.masked_fill(~keep.unsqueeze(-1), 0.0)
        pooled = _last_nonpad(encoded, keep.long())
        return encoded, pooled

    def _user_vector(self, uid: torch.Tensor, ids: torch.Tensor, kinds: torch.Tensor) -> torch.Tensor:
        hidden = self.input(ids, kinds)
        keep_click = kinds.eq(SATISFIED)
        keep_middle = kinds.eq(CLICK)
        keep_feedback = kinds.eq(DISSATISFIED)
        _, h_click = self._encode(self.enc_click, hidden, keep_click)
        H_middle, h_middle = self._encode(self.enc_middle, hidden, keep_middle)
        _, h_feedback = self._encode(self.enc_feedback, hidden, keep_feedback)
        middle_by_click = self.attn_click(H_middle, keep_middle, h_click)
        middle_by_feedback = self.attn_feedback(H_middle, keep_middle, h_feedback)
        components = torch.cat([
            self.user_embedding(uid), h_click, h_middle, h_feedback, middle_by_click, middle_by_feedback,
        ], dim=-1)
        return self.fuse(components)

    def score(self, uid, mixed_his, mixed_type, items):
        vector = self._user_vector(uid, mixed_his, mixed_type)
        return self._candidate_score(vector, self.input.item_embedding(items))


class DMT(MixedBaseline):
    """Deep Multifaceted Transformers adapted to questionnaire behaviors.

    Each of CLICK / SATISFIED / DISSATISFIED gets its own Transformer encoder
    over the mixed stream (with the other-type tokens masked out).  The
    candidate item embedding is used as a query attending over the encoded
    hidden state ("encode-decode"), yielding one vector per behavior.  The
    three vectors are then fused by a lightweight MMoE layer whose task tower
    count is reduced from the paper's two (CTR/CVR) to one (ranking).
    """

    _KINDS = (CLICK, SATISFIED, DISSATISFIED)

    def __init__(self, n_items: int, dim: int, max_len: int, dropout: float = 0.2,
                 num_heads: int = 2, num_blocks: int = 2, num_experts: int = 4) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("DMT dim must be divisible by num_heads")
        if num_experts < 1:
            raise ValueError("DMT num_experts must be positive")
        self.input = MixedInput(n_items, dim, max_len, dropout)
        self.encoders = nn.ModuleDict({
            str(kind): nn.TransformerEncoder(
                nn.TransformerEncoderLayer(dim, num_heads, dim_feedforward=dim * 4,
                                           dropout=dropout, batch_first=True),
                num_layers=num_blocks,
            ) for kind in self._KINDS
        })
        self.decoders = nn.ModuleDict({
            str(kind): nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            for kind in self._KINDS
        })
        bottom = dim * len(self._KINDS)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(bottom, dim), nn.GELU(), nn.Dropout(dropout)) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(bottom, num_experts)
        self.tower = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))

    def _encode_type(self, hidden: torch.Tensor, keep: torch.Tensor, kind: int) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoders[str(kind)](
            hidden.masked_fill(~keep.unsqueeze(-1), 0.0),
            src_key_padding_mask=_safe_padding(~keep),
        )
        encoded = encoded.masked_fill(~keep.unsqueeze(-1), 0.0)
        return encoded, keep

    def _decode(self, kind: int, query: torch.Tensor, memory: torch.Tensor,
                keep: torch.Tensor) -> torch.Tensor:
        # The candidate item embedding is the decoder query; the encoded
        # per-type sequence supplies the key/value.  ``_safe_padding`` guards
        # against all-padded rows by keeping one dummy position open, so
        # nn.MultiheadAttention never sees an entirely-masked query row.
        output, _ = self.decoders[str(kind)](
            query, memory, memory,
            key_padding_mask=_safe_padding(~keep), need_weights=False,
        )
        return output

    def score(self, uid, mixed_his, mixed_type, items):
        del uid
        hidden = self.input(mixed_his, mixed_type)
        item_vectors = self.input.item_embedding(items)
        if item_vectors.ndim == 2:
            item_vectors = item_vectors.unsqueeze(1)
        pooled = []
        for kind in self._KINDS:
            keep = mixed_type.eq(kind)
            encoded, keep_out = self._encode_type(hidden, keep, kind)
            pooled.append(self._decode(kind, item_vectors, encoded, keep_out))
        bottom = torch.cat(pooled, dim=-1)
        expert_outputs = torch.stack([expert(bottom) for expert in self.experts], dim=-2)
        gate_weights = torch.softmax(self.gate(bottom), dim=-1).unsqueeze(-1)
        mixture = (expert_outputs * gate_weights).sum(dim=-2)
        vector = self.tower(mixture)
        score = (vector * item_vectors).sum(-1)
        return score.squeeze(1) if score.size(1) == 1 else score


MULTIBEHAVIOR_MODELS.update({"DFN": DFN, "DMT": DMT})


def build_multibehavior_model(name: str, n_users: int, n_items: int, dim: int, feedback_len: int,
                               dropout: float, num_heads: int, num_blocks: int,
                               disentangle_weight: float, num_experts: int) -> MixedBaseline:
    if name == "GRU4RecM":
        return GRU4RecM(n_items, dim, feedback_len, dropout)
    if name == "SASRecM":
        return SASRecM(n_items, dim, feedback_len, dropout, num_heads, num_blocks)
    if name == "FMLPRecM":
        return FMLPRecM(n_items, dim, feedback_len, dropout, num_blocks)
    if name == "FeedRec":
        return FeedRec(n_items, dim, feedback_len, dropout, num_heads, num_blocks, disentangle_weight)
    if name == "DFN":
        return DFN(n_users, n_items, dim, feedback_len, dropout, num_heads, num_blocks)
    if name == "DMT":
        return DMT(n_items, dim, feedback_len, dropout, num_heads, num_blocks, num_experts)
    raise ValueError(f"Unknown multi-behavior model {name}; choose one of {sorted(MULTIBEHAVIOR_MODELS)}")
