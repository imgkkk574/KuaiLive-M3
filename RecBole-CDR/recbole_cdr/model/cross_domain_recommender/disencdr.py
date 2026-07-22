# -*- coding: utf-8 -*-
r"""
DisenCDR
################################################
Reference:
    Jiangxia Cao et al. "DisenCDR: Learning Disentangled Representations for
    Cross-Domain Recommendation." in SIGIR 2022.

Native RecBole-CDR port of the upstream standalone implementation
(https://github.com/WenjieWWJ/DisenCDR). The original trained on its own
``train.txt``/``test.txt`` triplets with a hand-written loop; this version
inherits :class:`CrossDomainRecommender` and runs under RecBole-CDR's trainer
so it shares the KLM3 ``.inter`` artifacts and ``run_tune.sh`` grid with the
other baselines (CMF, EMCDR, BiTGCF, MGCCDR, ...).

Disentanglement design
----------------------
Each user is represented as the sum of two parts:

  * **shared**   — a cross-domain latent drawn from a variational distribution
    fused across both domains (captures transferable preference).
  * **specific** — a domain-private variational representation (captures
    domain-unique preference).

Four single-domain VAE-GNNs (source/target × specific/share) plus one
cross-domain VAE-GNN produce these. A KL term aligns the fused shared
distribution with each domain's own shared distribution, disentangling the two.
The first ``warmup_epochs`` epochs train only the specific branches (KLD=0) to
stabilize the per-domain encoders before switching on disentanglement.
"""

import types

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole_cdr.model.cross_domain_recommender._disencdr_modules import (
    singleVBGE,
    crossVBGE,
)


def _cal_bpr_loss(pred):
    """pred: [B, 2] -> [pos, neg]. Standard BPR (matches MGCCDR)."""
    pos = pred[:, 0]
    neg = pred[:, 1]
    return -torch.log(torch.sigmoid(pos - neg) + 1e-8).mean()


class DisenCDR(CrossDomainRecommender):
    r"""Disentangled cross-domain recommendation via shared/specific VAE-GNNs.

    Trains with BPR on the target (live) domain. Shared/specific disentanglement
    is enforced through three KL terms controlled by ``beta``. A ``warmup_epochs``
    warm-up phase trains only the specific encoders.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(DisenCDR, self).__init__(config, dataset)

        self.config = config   # nn.Module doesn't auto-store it (BiTGCF-style)
        self.device = config['device']

        self.feature_dim = config['embedding_size']
        self.hidden_dim = config['embedding_size']   # keep equal (paper default)
        self.n_layers = config['n_layers']
        self.dropout = config['dropout']
        self.beta = config['beta']
        self.leakey = config['leakey']
        self.reg_weight = config['reg_weight']
        self.warmup_epochs = config['warmup_epochs']

        # Cross-domain fusion weight: source interactions / (source + target)
        # for each shared user. KLM3 has a fully-overlapping user space, so we
        # take a single global ratio (matches the upstream `rate()` scalar).
        src_inter = dataset.inter_matrix(form='coo', value_field=None,
                                         domain='source').astype(np.float32)
        tgt_inter = dataset.inter_matrix(form='coo', value_field=None,
                                         domain='target').astype(np.float32)
        src_deg = np.asarray(src_inter.sum(axis=1)).ravel()
        tgt_deg = np.asarray(tgt_inter.sum(axis=1)).ravel()
        denom = src_deg + tgt_deg
        valid = denom > 0
        rate = float(np.mean(src_deg[valid] / denom[valid])) if valid.any() else 0.5

        # Build the cfg namespace consumed by the ported sub-modules.
        self._gnn_cfg = types.SimpleNamespace(
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=self.dropout,
            leakey=self.leakey,
            rate=rate,
        )

        # Embeddings sized to the unified space (as in BiTGCF/MGCCDR).
        # Streamers (target items) live in [0, target_num_items);
        # videos    (source items) live in [target_num_items, total_num_items).
        self.source_user_embedding = nn.Embedding(self.total_num_users, self.feature_dim)
        self.target_user_embedding = nn.Embedding(self.total_num_users, self.feature_dim)
        self.source_item_embedding = nn.Embedding(self.total_num_items, self.feature_dim)
        self.target_item_embedding = nn.Embedding(self.total_num_items, self.feature_dim)
        self.source_user_embedding_share = nn.Embedding(self.total_num_users, self.feature_dim)
        self.target_user_embedding_share = nn.Embedding(self.total_num_users, self.feature_dim)

        # Shared-distribution projection (mirrors upstream share_mean/share_sigma).
        self.share_mean = nn.Linear(self.feature_dim + self.feature_dim, self.feature_dim)
        self.share_sigma = nn.Linear(self.feature_dim + self.feature_dim, self.feature_dim)

        # The 5 GNNs.
        self.source_specific_GNN = singleVBGE(self._gnn_cfg)
        self.source_share_GNN = singleVBGE(self._gnn_cfg)
        self.target_specific_GNN = singleVBGE(self._gnn_cfg)
        self.target_share_GNN = singleVBGE(self._gnn_cfg)
        self.share_GNN = crossVBGE(self._gnn_cfg)

        # Precomputed full-graph indices for all users/items (forward computes
        # the whole graph at once, as upstream does). Registered as buffers so
        # they follow model.to(device) and stay off the parameter list.
        self.register_buffer('_source_user_index', torch.arange(self.total_num_users))
        self.register_buffer('_target_user_index', torch.arange(self.total_num_users))
        self.register_buffer('_source_item_index', torch.arange(self.total_num_items))
        self.register_buffer('_target_item_index', torch.arange(self.total_num_items))

        # Build the four bipartite adjacency tensors (UV, VU per domain).
        self.source_UV, self.source_VU = self._bipartite_adj(src_inter)
        self.target_UV, self.target_VU = self._bipartite_adj(tgt_inter)
        self.source_UV = self.source_UV.to(self.device)
        self.source_VU = self.source_VU.to(self.device)
        self.target_UV = self.target_UV.to(self.device)
        self.target_VU = self.target_VU.to(self.device)

        # eval cache
        self.target_restore_user_e = None
        self.target_restore_item_e = None

        # warmup bookkeeping (advanced by DisenCDRTrainer each epoch)
        self._global_epoch = 0
        self.phase = 'BOTH'

        # Initialize, THEN zero out non-existent rows so the zeroing persists
        # past xavier init (MGCCDR/BiTGCF pattern).
        self.apply(xavier_normal_initialization)
        with torch.no_grad():
            # Source items (videos) occupy [target_num_items, total_num_items);
            # target items (streamers) occupy [0, target_num_items).
            # overlapped_num_items == 0 on KLM3 (photo vs author are distinct).
            self.source_item_embedding.weight[:self.target_num_items].fill_(0)
            self.target_item_embedding.weight[self.target_num_items:].fill_(0)
            # Users fully overlap on KLM3 (same user_id space + klm3_user.link),
            # so source/target user embeddings stay full-range. If a future
            # preprocessing yields partial user overlap, mirror the item
            # zero-out using overlapped_num_users.
        self.other_parameter_name = ['target_restore_user_e', 'target_restore_item_e']

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def _bipartite_adj(self, inter_matrix):
        """Build row-normalized UV (user×item) and VU (item×user) sparse tensors
        from a RecBole ``inter_matrix(domain=...)`` output. Mirrors the upstream
        ``GraphMaker.preprocess`` + ``normalize``."""
        coo = inter_matrix.tocoo().astype(np.float32)
        n_user, n_item = coo.shape

        # UV: user -> item, row-normalized over items (per user)
        uv = sp.coo_matrix((np.ones(coo.nnz), (coo.row, coo.col)),
                           shape=(n_user, n_item), dtype=np.float32)
        uv = self._row_normalize(uv)
        uv_t = self._to_torch_sparse(uv)

        # VU: item -> user, row-normalized over users (per item) = UV.T row-norm
        vu = sp.coo_matrix((np.ones(coo.nnz), (coo.col, coo.row)),
                           shape=(n_item, n_user), dtype=np.float32)
        vu = self._row_normalize(vu)
        vu_t = self._to_torch_sparse(vu)
        return uv_t, vu_t

    @staticmethod
    def _row_normalize(mx):
        rowsum = np.array(mx.sum(axis=1)).ravel()
        r_inv = np.power(rowsum, -1, where=rowsum > 0)
        r_inv[np.isinf(r_inv)] = 0.0
        return sp.diags(r_inv).dot(mx).tocoo()

    @staticmethod
    def _to_torch_sparse(sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        return torch.sparse_coo_tensor(indices, values, torch.Size(sparse_mx.shape))

    # ------------------------------------------------------------------
    # Variational helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _kld_gauss(mu_1, logsigma_1, mu_2, logsigma_2):
        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_1))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_2))
        q_target = Normal(mu_1, sigma_1)
        q_context = Normal(mu_2, sigma_2)
        return kl_divergence(q_target, q_context).mean(dim=0).sum()

    def reparameters(self, mean, logstd):
        sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        gaussian_noise = torch.randn(mean.size(0), self.hidden_dim, device=mean.device)
        if self.share_mean.training:
            sampled_z = gaussian_noise * torch.exp(sigma) + mean
        else:
            sampled_z = mean
        kld_loss = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, (1 - self.beta) * kld_loss

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, source_UV, source_VU, target_UV, target_VU):
        source_user = self.source_user_embedding(self._source_user_index)
        target_user = self.target_user_embedding(self._target_user_index)
        source_item = self.source_item_embedding(self._source_item_index)
        target_item = self.target_item_embedding(self._target_item_index)
        source_user_share = self.source_user_embedding_share(self._source_user_index)
        target_user_share = self.target_user_embedding_share(self._target_user_index)

        # domain-specific branches (each yields user, item)
        source_learn_specific_user, source_learn_specific_item = self.source_specific_GNN(
            source_user, source_item, source_UV, source_VU)
        target_learn_specific_user, target_learn_specific_item = self.target_specific_GNN(
            target_user, target_item, target_UV, target_VU)

        # per-domain shared-user (mean, sigma)
        source_user_mean, source_user_sigma = self.source_share_GNN.forward_user_share(
            source_user_share, source_UV, source_VU)
        target_user_mean, target_user_sigma = self.target_share_GNN.forward_user_share(
            target_user_share, target_UV, target_VU)

        # cross-domain fused shared distribution
        mean, sigma = self.share_GNN(
            source_user_share, target_user_share,
            source_UV, source_VU, target_UV, target_VU)

        user_share, share_kld_loss = self.reparameters(mean, sigma)
        source_share_kld = self._kld_gauss(mean, sigma, source_user_mean, source_user_sigma)
        target_share_kld = self._kld_gauss(mean, sigma, target_user_mean, target_user_sigma)

        self.kld_loss = share_kld_loss + self.beta * source_share_kld + \
                        self.beta * target_share_kld

        # final user = shared + specific; items use only their specific branch
        source_learn_user = user_share + source_learn_specific_user
        target_learn_user = user_share + target_learn_specific_user
        return source_learn_user, source_learn_specific_item, \
               target_learn_user, target_learn_specific_item

    def warmup_forward(self, source_UV, source_VU, target_UV, target_VU):
        """Specific-only forward; KLD disabled. Used during the warm-up phase."""
        source_user = self.source_user_embedding(self._source_user_index)
        target_user = self.target_user_embedding(self._target_user_index)
        source_item = self.source_item_embedding(self._source_item_index)
        target_item = self.target_item_embedding(self._target_item_index)

        source_learn_specific_user, source_learn_specific_item = self.source_specific_GNN(
            source_user, source_item, source_UV, source_VU)
        target_learn_specific_user, target_learn_specific_item = self.target_specific_GNN(
            target_user, target_item, target_UV, target_VU)
        self.kld_loss = 0
        return source_learn_specific_user, source_learn_specific_item, \
               target_learn_specific_user, target_learn_specific_item

    # ------------------------------------------------------------------
    # RecBole interface
    # ------------------------------------------------------------------
    def calculate_loss(self, interaction):
        self.init_restore_e()

        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_neg = interaction[self.TARGET_NEG_ITEM_ID]

        if self.warmup_epochs > 0 and self._global_epoch < self.warmup_epochs:
            src_u, src_i, tgt_u, tgt_i = self.warmup_forward(
                self.source_UV, self.source_VU, self.target_UV, self.target_VU)
            kld = 0
        else:
            src_u, src_i, tgt_u, tgt_i = self.forward(
                self.source_UV, self.source_VU, self.target_UV, self.target_VU)
            kld = self.kld_loss

        u_e = tgt_u[target_user]
        pos_e = tgt_i[target_item]
        neg_e = tgt_i[target_neg]
        pred = torch.stack([torch.mul(u_e, pos_e).sum(dim=1),
                            torch.mul(u_e, neg_e).sum(dim=1)], dim=1)
        bpr_loss = _cal_bpr_loss(pred)

        # L2 reg on the embeddings used this batch
        reg_loss = (u_e.norm(2) + pos_e.norm(2) + neg_e.norm(2)) / target_user.shape[0]
        return bpr_loss + self.reg_weight * reg_loss + kld

    def predict(self, interaction):
        _, _, tgt_u, tgt_i = self._forward_eval()
        u_e = tgt_u[interaction[self.TARGET_USER_ID]]
        i_e = tgt_i[interaction[self.TARGET_ITEM_ID]]
        return torch.mul(u_e, i_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        restore_user_e, restore_item_e = self.get_restore_e()
        u_e = restore_user_e[user]
        i_e = restore_item_e[:self.target_num_items]
        return torch.matmul(u_e, i_e.t()).view(-1)

    def _forward_eval(self):
        """Forward for inference: warmup-agnostic (always uses the full model).
        The graph convolutions are deterministic in eval mode (no dropout
        reparam noise for the *specific* items; the shared branch still
        reparameterizes but uses mean in eval)."""
        return self.forward(self.source_UV, self.source_VU,
                            self.target_UV, self.target_VU)

    def init_restore_e(self):
        if self.target_restore_user_e is not None or self.target_restore_item_e is not None:
            self.target_restore_user_e, self.target_restore_item_e = None, None

    def get_restore_e(self):
        if self.target_restore_user_e is None or self.target_restore_item_e is None:
            _, _, tgt_u, tgt_i = self._forward_eval()
            self.target_restore_user_e = tgt_u
            self.target_restore_item_e = tgt_i
        return self.target_restore_user_e, self.target_restore_item_e

    # ------------------------------------------------------------------
    # Phase / epoch hooks (called by DisenCDRTrainer)
    # ------------------------------------------------------------------
    def set_phase(self, phase):
        self.phase = phase

    def set_epoch(self, epoch):
        """Advance the global epoch counter for warm-up gating."""
        self._global_epoch = epoch
