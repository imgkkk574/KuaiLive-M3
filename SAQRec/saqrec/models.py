from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _last_nonpad(hidden: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    lengths = ids.ne(0).sum(dim=1).clamp_min(1) - 1
    return hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]


class Encoder(nn.Module):
    def __init__(self, n_items: int, dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_items, dim, padding_idx=0)
        self.position = nn.Embedding(512, dim)
        layer = nn.TransformerEncoderLayer(dim, nhead=2, dim_feedforward=dim * 2,
                                           dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        x = self.embedding(ids) + self.position(positions)
        padding = ids.eq(0)
        # Transformer attention is undefined for an entirely padded sequence.
        # Empty questionnaire histories are valid in KLM3, so expose one zero
        # token internally and mask the resulting representation back to zero.
        safe_padding = padding.clone()
        safe_padding[padding.all(dim=1), 0] = False
        hidden = self.transformer(x, src_key_padding_mask=safe_padding)
        hidden = hidden.masked_fill(ids.eq(0).unsqueeze(-1), 0.0)
        return hidden, _last_nonpad(hidden, ids)


class BaseRec(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, dim)
        self.encoder = Encoder(n_items, dim, dropout)
        self.item_embedding = self.encoder.embedding
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, 2))

    def user_vector(self, uid: torch.Tensor, rec_his: torch.Tensor) -> torch.Tensor:
        _, history = self.encoder(rec_his)
        user = self.user_embedding(uid)
        weights = torch.softmax(self.gate(torch.cat([history, user], dim=-1)), dim=-1)
        return weights[:, :1] * history + weights[:, 1:] * user

    def score(self, uid: torch.Tensor, rec_his: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user = self.user_vector(uid, rec_his)
        item = self.item_embedding(items)
        return (user.unsqueeze(1) * item).sum(dim=-1) if items.ndim == 2 else (user * item).sum(dim=-1)

    def loss(self, batch) -> torch.Tensor:
        pos = self.score(batch["uid"], batch["rec_his"], batch["pos"])
        neg = self.score(batch["uid"], batch["rec_his"], batch["neg"])
        return F.softplus(-pos).mean() + F.softplus(neg).mean()


class PropensityModel(nn.Module):
    def __init__(self, base: BaseRec, hidden: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding.from_pretrained(base.user_embedding.weight.detach().clone(), freeze=True)
        self.item_embedding = nn.Embedding.from_pretrained(base.item_embedding.weight.detach().clone(), freeze=True,
                                                            padding_idx=0)
        dim = self.user_embedding.embedding_dim
        self.mlp = nn.Sequential(nn.Linear(dim * 2, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, uid: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.user_embedding(uid), self.item_embedding(item)], dim=-1)
        return self.mlp(x).squeeze(-1)


class SatisfactionModel(PropensityModel):
    def ips_loss(self, uid, item, label, propensity, clamp: float) -> torch.Tensor:
        logits = self(uid, item)
        per_sample = F.binary_cross_entropy_with_logits(logits, label, reduction="none")
        return (per_sample / propensity.detach().sigmoid().clamp_min(clamp)).mean()


class SAQRec(nn.Module):
    """SAQRec with SID, MLSE and adaptive satisfaction-label correction."""
    def __init__(self, n_users: int, n_items: int, teacher: SatisfactionModel,
                 dim: int = 64, dropout: float = 0.2, num_interest: int = 8) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, dim)
        self.item_embedding = nn.Embedding(n_items, dim, padding_idx=0)
        self.rec_encoder = Encoder(n_items, dim, dropout)
        self.satis_encoder = Encoder(n_items, dim, dropout)
        self.dissatis_encoder = Encoder(n_items, dim, dropout)
        self.teacher = teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.num_interest = num_interest
        self.sid_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, 2))
        self.final_gate = nn.Sequential(nn.Linear(dim * 5, dim), nn.ReLU(), nn.Linear(dim, 5))
        self.satisfaction_projection = nn.Linear(dim, dim, bias=False)

    def initialize_from_base(self, base: BaseRec) -> None:
        """Warm-start the main recommender exactly once from Base embeddings."""
        with torch.no_grad():
            self.user_embedding.weight.copy_(base.user_embedding.weight)
            self.item_embedding.weight.copy_(base.item_embedding.weight)
            for encoder in (self.rec_encoder, self.satis_encoder, self.dissatis_encoder):
                encoder.embedding.weight.copy_(base.item_embedding.weight)

    @staticmethod
    def _attention(values: torch.Tensor, key_ids: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        scores = (values * query.unsqueeze(1)).sum(dim=-1)
        scores = scores.masked_fill(key_ids.eq(0), -1e9)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(key_ids.ne(0), weights, torch.zeros_like(weights))
        return (values * weights.unsqueeze(-1)).sum(dim=1)

    def _mlse(self, uid: torch.Tensor, rec_ids: torch.Tensor, rec_hidden: torch.Tensor) -> torch.Tensor:
        """Batch implementation of the paper's satisfaction-level grouping.

        The previous implementation iterated once per user in Python.  With a
        KLM3 batch of 4,096 this dominated wall time.  Sorting and group-wise
        softmaxes below are mathematically equivalent to the previous
        ``argsort`` + ``tensor_split`` formulation, but run as batched tensor
        operations on the accelerator.
        """
        with torch.no_grad():
            teacher_scores = self.teacher(uid.unsqueeze(1).expand_as(rec_ids).reshape(-1), rec_ids.reshape(-1))
            teacher_scores = teacher_scores.sigmoid().reshape_as(rec_ids)
        valid = rec_ids.ne(0)
        batch, length, dim = rec_hidden.shape
        valid_count = valid.sum(dim=1)
        group_count = valid_count.clamp(max=self.num_interest)
        sortable = teacher_scores.masked_fill(~valid, -float("inf"))
        order = torch.argsort(sortable, dim=1, descending=True)
        sorted_scores = sortable.gather(1, order)
        sorted_hidden = rec_hidden.gather(1, order.unsqueeze(-1).expand(-1, -1, dim))
        sorted_valid = torch.arange(length, device=rec_ids.device).unsqueeze(0) < valid_count.unsqueeze(1)

        # torch.tensor_split(n, g) puts the larger groups first.  For sorted
        # ranks r this is exactly floor(r * g / n), where g=min(NI, n).
        ranks = torch.arange(length, device=rec_ids.device).unsqueeze(0).expand(batch, -1)
        group_index = torch.div(ranks * group_count.unsqueeze(1), valid_count.clamp_min(1).unsqueeze(1),
                                rounding_mode="floor").clamp_max(self.num_interest - 1)
        slots = torch.arange(self.num_interest, device=rec_ids.device).view(1, 1, -1)
        group_valid = slots.squeeze(1) < group_count.unsqueeze(1)
        membership = sorted_valid.unsqueeze(-1) & group_valid.unsqueeze(1) & (group_index.unsqueeze(-1) == slots)

        within_logits = sorted_scores.unsqueeze(-1).masked_fill(~membership, -float("inf"))
        within_weights = torch.softmax(within_logits, dim=1)
        within_weights = torch.where(membership, within_weights, torch.zeros_like(within_weights))
        group_vectors = torch.einsum("blg,bld->bgd", within_weights, sorted_hidden)
        group_scores = group_vectors.mean(dim=-1).masked_fill(~group_valid, -float("inf"))
        group_weights = torch.softmax(group_scores, dim=1)
        group_weights = torch.where(group_valid, group_weights, torch.zeros_like(group_weights))
        return torch.einsum("bg,bgd->bd", group_weights, group_vectors)

    def user_vector(self, uid, rec_his, satis_his, dissatis_his):
        rec_hidden, rec = self.rec_encoder(rec_his)
        _, satis = self.satis_encoder(satis_his)
        _, dissatis = self.dissatis_encoder(dissatis_his)
        satis_from_rec = self._attention(rec_hidden, rec_his, satis)
        dissatis_from_rec = self._attention(rec_hidden, rec_his, dissatis)
        g_pos = torch.softmax(self.sid_gate(torch.cat([satis, satis_from_rec], -1)), -1)
        g_neg = torch.softmax(self.sid_gate(torch.cat([dissatis, dissatis_from_rec], -1)), -1)
        pos = g_pos[:, :1] * satis + g_pos[:, 1:] * satis_from_rec
        neg = g_neg[:, :1] * dissatis + g_neg[:, 1:] * dissatis_from_rec
        multi = self._mlse(uid, rec_his, rec_hidden)
        user = self.user_embedding(uid)
        components = torch.stack([pos, neg, rec, multi, user], dim=1)
        weights = torch.softmax(self.final_gate(components.flatten(1)), dim=-1)
        return (components * weights.unsqueeze(-1)).sum(dim=1)

    def scores_from_vector(self, vector: torch.Tensor, items: torch.Tensor):
        item = self.item_embedding(items)
        click = (vector.unsqueeze(1) * item).sum(-1) if items.ndim == 2 else (vector * item).sum(-1)
        satis_item = self.satisfaction_projection(item)
        satisfaction = ((vector.unsqueeze(1) * satis_item).sum(-1) if items.ndim == 2
                        else (vector * satis_item).sum(-1))
        return click, satisfaction

    def scores(self, batch, items: torch.Tensor):
        vector = self.user_vector(batch["uid"], batch["rec_his"], batch["satis_his"], batch["dissatis_his"])
        return self.scores_from_vector(vector, items)

    @staticmethod
    def _beta_half_cdf(x: torch.Tensor) -> torch.Tensor:
        # Exact CDF of Beta(1/2, 1/2), avoiding a SciPy dependency.
        return (2.0 / math.pi) * torch.asin(torch.sqrt(x.clamp(0.0, 1.0)))

    def loss(self, batch, satisfaction_weight: float, correction_after: int, epoch: int):
        # Positive and negative candidates share one user history.  Encoding
        # it once avoids a complete duplicate set of three Transformers and
        # MLSE teacher/grouping work every training step.
        vector = self.user_vector(batch["uid"], batch["rec_his"], batch["satis_his"], batch["dissatis_his"])
        pos_click, pos_sat = self.scores_from_vector(vector, batch["pos"])
        neg_click, neg_sat = self.scores_from_vector(vector, batch["neg"])
        click_loss = F.softplus(-pos_click).mean() + F.softplus(neg_click).mean()
        with torch.no_grad():
            teacher_pos = self.teacher(batch["uid"], batch["pos"]).sigmoid()
            teacher_neg = self.teacher(batch["uid"].unsqueeze(1).expand_as(batch["neg"]).reshape(-1),
                                        batch["neg"].reshape(-1)).sigmoid().reshape_as(neg_sat)
        all_pred = torch.cat([pos_sat.unsqueeze(1), neg_sat], dim=1)
        all_teacher = torch.cat([teacher_pos.unsqueeze(1), teacher_neg], dim=1)
        if epoch >= correction_after:
            # This is a target-correction step, not an additional learnable
            # path.  In particular, Beta(1/2, 1/2)'s CDF has unbounded
            # derivatives at 0 and 1; min/max normalization produces those
            # endpoints in nearly every batch.  Keeping this graph attached
            # can therefore turn the first corrected epoch into NaNs.
            with torch.no_grad():
                raw = F.binary_cross_entropy_with_logits(all_pred, all_teacher, reduction="none")
                normalized = (raw - raw.min()) / (raw.max() - raw.min()).clamp_min(1e-8)
                # Keep the analytic CDF away from its endpoint singularities.
                lam = self._beta_half_cdf(normalized.clamp(1e-6, 1.0 - 1e-6))
                labels = lam * all_teacher + (1.0 - lam) * all_pred.sigmoid()
        else:
            labels = all_teacher
        sat_loss = F.binary_cross_entropy_with_logits(all_pred, labels)
        return click_loss + satisfaction_weight * sat_loss
