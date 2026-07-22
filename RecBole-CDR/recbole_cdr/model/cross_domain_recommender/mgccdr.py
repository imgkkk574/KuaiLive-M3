# -*- coding: utf-8 -*-
r"""
MGCCDR
################################################
Reference:
    Changle Qu et al. "Bridging Short Videos and Streamers with Multi-Graph
    Contrastive Learning for Live Streaming Recommendation." in SIGIR 2025.

Adapted as a native RecBole-CDR model. The three bipartite graphs map onto KLM3 as:
    U-S (user-streamer): target domain (live) interaction matrix
    U-V (user-video)   : source domain (photo) interaction matrix
    S-V (streamer-video): author_id x photo_id ownership, loaded from an external
                          link file (config 'sv_link_file_path'); NOT via RecBole's
                          item_link mechanism, to avoid polluting the unified item
                          ID space (streamer and video are distinct entity types).
"""

import os

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


def _cal_bpr_loss(pred):
    # pred: [B, 2] -> [pos, neg]
    pos = pred[:, 0]
    neg = pred[:, 1]
    loss = -torch.log(torch.sigmoid(pos - neg) + 1e-8)
    return torch.mean(loss)


class MGCCDR(CrossDomainRecommender):
    r"""Multi-Graph Contrastive Cross-Domain Recommendation for live streaming.

    Builds three bipartite graphs (U-S, U-V, S-V), propagates embeddings on each
    to obtain three views of users and streamers, fuses them via a per-node
    attention over the three views (MGA), and trains with BPR loss on the target
    (live) domain. An optional InfoNCE contrastive loss between clean and
    noise-augmented representations is added when ``c_lambda > 0``.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(MGCCDR, self).__init__(config, dataset)

        self.config = config   # BiTGCF-style: keep a ref; nn.Module won't auto-store it
        self.device = config['device']
        self.latent_dim = config['embedding_size']
        self.n_layers = config['num_layers']
        self.alpha = config['alpha']           # weight of U-S view (target) in rec loss
        self.c_lambda = config['c_lambda']
        self.c_temp = config['c_temp']
        self.reg_weight = config['reg_weight']

        # noise augmentation ratios (0 disables the view's noise)
        self.eps_dict = {
            'US': config['US_ratio'],
            'UV': config['UV_ratio'],
            'SV': config['SV_ratio'],
        }

        # Embeddings sized to the unified space (as in BiTGCF). Streamers live in the
        # target-item segment [0, target_num_items); videos in the source-item segment.
        self.user_embedding = nn.Embedding(self.total_num_users, self.latent_dim)
        self.streamer_embedding = nn.Embedding(self.total_num_items, self.latent_dim)
        self.video_embedding = nn.Embedding(self.total_num_items, self.latent_dim)

        # Attention fusion: scores 3 views per node from [key || query]
        self.fusion_lin = nn.Linear(self.latent_dim * 2, 1)
        nn.init.xavier_uniform_(self.fusion_lin.weight)

        # ---- Build the three graphs ----
        us_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='target').astype(np.float32)
        uv_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='source').astype(np.float32)
        sv_matrix = self._build_sv_matrix(dataset)

        self.US_prop = self._propagation_graph(us_matrix).to(self.device)   # (U+S, U+S)
        self.UV_prop = self._propagation_graph(uv_matrix).to(self.device)    # (U+V, U+V)
        self.SV_prop = self._propagation_graph(sv_matrix).to(self.device)    # (S+V, S+V)
        self.SV_agg = self._aggregation_graph(sv_matrix).to(self.device)     # (S, V) row-norm
        self.UV_agg = self._aggregation_graph(uv_matrix).to(self.device)     # (U, V) row-norm

        # eval cache
        self.target_restore_user_e = None
        self.target_restore_item_e = None

        # initialize, THEN zero out non-existent rows so the zeroing persists.
        self.apply(xavier_normal_initialization)
        with torch.no_grad():
            self.streamer_embedding.weight[self.target_num_items:].fill_(0)
            self.video_embedding.weight[self.overlapped_num_items:self.target_num_items].fill_(0)
        self.other_parameter_name = ['target_restore_user_e', 'target_restore_item_e']

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    @staticmethod
    def _propagation_graph(bipartite):
        """Symmetric Laplacian-normalized adjacency D^{-1/2} [[0,B],[B^T,0]] D^{-1/2}."""
        n_u, n_v = bipartite.shape
        A = sp.bmat([[sp.csr_matrix((n_u, n_u)), bipartite.tocsr()],
                     [bipartite.T.tocsr(), sp.csr_matrix((n_v, n_v))]], format='coo')
        deg = np.array(A.sum(axis=1)).ravel() + 1e-8
        d_inv_sqrt = sp.diags(np.power(deg, -0.5))
        L = d_inv_sqrt @ A @ d_inv_sqrt
        return _to_torch_sparse(L.tocoo())

    @staticmethod
    def _aggregation_graph(bipartite):
        """Row-normalized bipartite D_r^{-1} B, used to project one node type's
        features onto the other (e.g. video -> streamer via S-V aggregation)."""
        row = bipartite.tocsr()
        deg = np.array(row.sum(axis=1)).ravel() + 1e-8
        agg = sp.diags(1.0 / deg) @ row
        return _to_torch_sparse(agg.tocoo())

    def _build_sv_matrix(self, dataset):
        """Build the (num_streamers x num_videos) sparse ownership matrix from the
        external sv link file. Maps raw author_id/photo_id tokens to unified IDs.

        Streamer unified IDs == target-item IDs in [0, target_num_items).
        Video unified IDs == source-item IDs in [target_num_items, total_num_items)
        (offset by target_num_items, since overlap_item == 0 on KLM3).
        """
        sv_path = self.config['sv_link_file_path']
        if sv_path is None:
            raise ValueError("MGCCDR requires config['sv_link_file_path'] pointing to klm3_sv.link")
        # sv_link_file_path is relative to the RecBole-CDR working dir (e.g. 'dataset/klm3_sv.link').
        if not os.path.exists(sv_path):
            raise FileNotFoundError(f"MGCCDR: sv link file not found: {sv_path}")

        import csv
        pairs = []
        with open(sv_path, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            for row in reader:
                if len(row) < 2:
                    continue
                pairs.append((row[0], row[1]))

        tgt_field = self.TARGET_ITEM_ID   # 'target_author_id'
        src_field = self.SOURCE_ITEM_ID   # 'source_photo_id'
        author2id = dataset.target_domain_dataset.field2token_id[tgt_field]
        photo2id = dataset.source_domain_dataset.field2token_id[src_field]

        rows, cols = [], []
        for author_tok, photo_tok in pairs:
            if author_tok in author2id and photo_tok in photo2id:
                sid = author2id[author_tok]              # streamer unified id in [0, target_num_items)
                vid = photo2id[photo_tok]                 # video unified id
                rows.append(sid)
                cols.append(vid)
        if not rows:
            self.logger.warning("MGCCDR: empty S-V graph after token mapping "
                                "(check author_id/photo_id coverage).")
        data = np.ones(len(rows), dtype=np.float32)
        return sp.coo_matrix((data, (rows, cols)),
                             shape=(self.total_num_items, self.total_num_items))

    # ------------------------------------------------------------------
    # Propagation / aggregation
    # ------------------------------------------------------------------
    def _propagate(self, graph, feat_a, feat_b, graph_type, test):
        """LightGCN-style propagation on the symmetric graph, with optional noise.

        Streaming mean over layers to keep peak memory at ~2 tensors instead of
        O(num_layers) — critical for KLM3's ~3M-item graphs.
        """
        features = torch.cat((feat_a, feat_b), dim=0)
        eps = self.eps_dict[graph_type]
        # streaming mean: accumulator = (ego + sum of layer outputs) / (n_layers + 1)
        accum = features.clone()
        for _ in range(self.n_layers):
            features = torch.sparse.mm(graph, features)
            if not test and eps > 0:
                noise = torch.rand_like(features)
                features.add_(torch.sign(features) * F.normalize(noise, dim=-1) * eps)
            accum = accum + F.normalize(features, p=2, dim=1)
        all_feats = accum / (self.n_layers + 1)
        a_out, b_out = torch.split(all_feats, (feat_a.shape[0], feat_b.shape[0]), dim=0)
        return a_out, b_out

    def _aggregate(self, agg_graph, node_feat, graph_type, test):
        """Project a node type's features onto the other via row-normalized bipartite."""
        out = torch.sparse.mm(agg_graph, node_feat)
        eps = self.eps_dict[graph_type]
        if not test and eps > 0:
            noise = torch.rand_like(out)
            out = out + torch.sign(out) * F.normalize(noise, dim=-1) * eps
        return out

    def get_multi_modal_representations(self, test=False):
        us_u, us_s = self._propagate(self.US_prop, self.user_embedding.weight,
                                     self.streamer_embedding.weight, 'US', test)
        uv_u, uv_v = self._propagate(self.UV_prop, self.user_embedding.weight,
                                     self.video_embedding.weight, 'UV', test)
        uv_s = self._aggregate(self.SV_agg, uv_v, 'SV', test)            # video -> streamer

        sv_s, sv_v = self._propagate(self.SV_prop, self.streamer_embedding.weight,
                                     self.video_embedding.weight, 'SV', test)
        sv_u = self._aggregate(self.UV_agg, sv_v, 'UV', test)            # video -> user

        users_views = [us_u, uv_u, sv_u]
        streamers_views = [us_s, uv_s, sv_s]
        return users_views, streamers_views

    def mga(self, views):
        """Fuse 3 views via per-node attention. views: list of [N, D] tensors."""
        key = views[0].unsqueeze(1).expand(-1, len(views), -1)          # [N, 3, D]
        query = torch.stack(views, dim=1)                                # [N, 3, D]
        attn = self.fusion_lin(torch.cat([key, query], dim=2)).softmax(dim=1)  # [N, 3, 1]
        fused = (attn * query).sum(dim=1)                                # [N, D]
        return fused

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def cal_c_loss(self, pos, aug):
        pos = F.normalize(pos[:, 0, :] if pos.dim() == 3 else pos, p=2, dim=1)
        aug = F.normalize(aug[:, 0, :] if aug.dim() == 3 else aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1)
        ttl_score = torch.matmul(pos, aug.t())
        pos_score = torch.exp(pos_score / self.c_temp)
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), dim=1)
        return -torch.mean(torch.log(pos_score / ttl_score + 1e-8))

    # ------------------------------------------------------------------
    # RecBole interface
    # ------------------------------------------------------------------
    def calculate_loss(self, interaction):
        self.init_restore_e()

        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_neg = interaction[self.TARGET_NEG_ITEM_ID]

        users_views, streamers_views = self.get_multi_modal_representations(test=False)
        user_rep = self.mga(users_views)
        streamer_rep = self.mga(streamers_views)

        u_e = user_rep[target_user]
        pos_e = streamer_rep[target_item]
        neg_e = streamer_rep[target_neg]
        pred = torch.stack([torch.mul(u_e, pos_e).sum(dim=1),
                            torch.mul(u_e, neg_e).sum(dim=1)], dim=1)
        bpr_loss = _cal_bpr_loss(pred)

        # L2 reg on the embeddings actually used this batch
        reg_loss = (user_rep[target_user].norm(2) + pos_e.norm(2) + neg_e.norm(2)) / target_user.shape[0]
        loss = bpr_loss + self.reg_weight * reg_loss

        # Contrastive loss: clean vs. a second noise-augmented forward
        if self.c_lambda > 0:
            u_aug_views, s_aug_views = self.get_multi_modal_representations(test=False)
            u_aug = self.mga(u_aug_views)
            s_aug = self.mga(s_aug_views)
            c_loss = (self.cal_c_loss(user_rep[target_user], u_aug[target_user]) +
                      self.cal_c_loss(streamer_rep[target_item], s_aug[target_item])) / 2.0
            loss = loss + self.c_lambda * c_loss

        return loss

    def predict(self, interaction):
        users_views, streamers_views = self.get_multi_modal_representations(test=True)
        user_rep = self.mga(users_views)
        streamer_rep = self.mga(streamers_views)
        u_e = user_rep[interaction[self.TARGET_USER_ID]]
        i_e = streamer_rep[interaction[self.TARGET_ITEM_ID]]
        return torch.mul(u_e, i_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        restore_user_e, restore_item_e = self.get_restore_e()
        u_e = restore_user_e[user]
        i_e = restore_item_e[:self.target_num_items]
        scores = torch.matmul(u_e, i_e.t())
        return scores.view(-1)

    def init_restore_e(self):
        if self.target_restore_user_e is not None or self.target_restore_item_e is not None:
            self.target_restore_user_e, self.target_restore_item_e = None, None

    def get_restore_e(self):
        if self.target_restore_user_e is None or self.target_restore_item_e is None:
            users_views, streamers_views = self.get_multi_modal_representations(test=True)
            self.target_restore_user_e = self.mga(users_views)
            self.target_restore_item_e = self.mga(streamers_views)
        return self.target_restore_user_e, self.target_restore_item_e


def _to_torch_sparse(coo):
    coo = coo.tocoo() if hasattr(coo, 'tocoo') else coo
    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
    values = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(indices, values, torch.Size(coo.shape))
