# -*- coding: utf-8 -*-
r"""
DisenCDR GNN sub-modules
################################################
Ported (and cleaned up) from the original DisenCDR release
(Cao et al., "DisenCDR: Learning Disentangled Representations for Cross-Domain
Recommendation", SIGIR 2022): model/GCN.py, model/singleVBGE.py, model/crossVBGE.py.

Changes vs. upstream:
  - Config access switched from ``opt["key"]`` dict to attribute fields on a
    plain ``argparse.Namespace``-style object (``cfg``) passed in at construction.
  - Removed ``.cuda()`` hard-coding; device is inferred from the input tensor
    (``torch.randn(..., device=mean.device)``) so the module runs on CPU too.
  - Dropped the unused ``VGAE`` class (the original ``GCN.py`` referenced
    ``Normal``/``kl_divergence`` inside it without importing them — dead code).
  - GCN now takes an explicit ``feature_dim``/``hidden_dim``/``dropout``/``alpha``
    instead of reading a global ``opt``.

These sub-modules operate on **per-domain** bipartite graphs (UV: user->item,
VU: item->user, both row-normalized sparse tensors). They are domain-agnostic;
the top-level ``DisenCDR`` model owns 4 singleVBGE + 1 crossVBGE instances and
feeds each the appropriate domain's adjacency.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence


# ----------------------------------------------------------------------
# Graph Convolution
# ----------------------------------------------------------------------
class GraphConvolution(nn.Module):
    """Single-layer sparse graph convolution: ``out = spmm(adj, input @ W) + b``."""

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)          # adj is a torch sparse tensor
        if self.bias is not None:
            return output + self.bias
        return output


class GCN(nn.Module):
    """One hop of graph convolution + LeakyReLU. Used to build the DGCN layers."""

    def __init__(self, nfeat, nhid, dropout, alpha):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        x = self.leakyrelu(self.gc1(x, adj))
        return x


# ----------------------------------------------------------------------
# Single-domain DGCN-VAE (specific OR share branch)
# ----------------------------------------------------------------------
class _DGCNLayer(nn.Module):
    """DGCN hidden layer: two-hop propagation on the bipartite graph + residual.

    ``User_ho = gc3(gc1(user, VU), UV)`` (user -> item -> user)
    ``Item_ho = gc4(gc2(item, UV), VU)`` (item -> user -> item)
    Then concat with the ego feature and project back to ``feature_dim``.
    """

    def __init__(self, cfg):
        super(_DGCNLayer, self).__init__()
        self.cfg = cfg
        self.gc1 = GCN(cfg.feature_dim, cfg.hidden_dim, cfg.dropout, cfg.leakey)
        self.gc2 = GCN(cfg.feature_dim, cfg.hidden_dim, cfg.dropout, cfg.leakey)
        self.gc3 = GCN(cfg.hidden_dim, cfg.feature_dim, cfg.dropout, cfg.leakey)
        self.gc4 = GCN(cfg.hidden_dim, cfg.feature_dim, cfg.dropout, cfg.leakey)
        self.user_union = nn.Linear(cfg.feature_dim + cfg.feature_dim, cfg.feature_dim)
        self.item_union = nn.Linear(cfg.feature_dim + cfg.feature_dim, cfg.feature_dim)

    def forward(self, ufea, vfea, UV_adj, VU_adj):
        User_ho = self.gc3(self.gc1(ufea, VU_adj), UV_adj)
        Item_ho = self.gc4(self.gc2(vfea, UV_adj), VU_adj)
        User = self.user_union(torch.cat((User_ho, ufea), dim=1))
        Item = self.item_union(torch.cat((Item_ho, vfea), dim=1))
        return F.relu(User), F.relu(Item)

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        User_ho = self.gc3(self.gc1(ufea, VU_adj), UV_adj)
        User = self.user_union(torch.cat((User_ho, ufea), dim=1))
        return F.relu(User)


class _LastLayer(nn.Module):
    """Final DGCN layer producing variational (mean, logstd) for user & item."""

    def __init__(self, cfg):
        super(_LastLayer, self).__init__()
        self.cfg = cfg
        self.gc1 = GCN(cfg.feature_dim, cfg.hidden_dim, cfg.dropout, cfg.leakey)
        self.gc2 = GCN(cfg.feature_dim, cfg.hidden_dim, cfg.dropout, cfg.leakey)
        self.gc3_mean = GCN(cfg.hidden_dim, cfg.feature_dim, cfg.dropout, cfg.leakey)
        self.gc3_logstd = GCN(cfg.hidden_dim, cfg.feature_dim, cfg.dropout, cfg.leakey)
        self.gc4_mean = GCN(cfg.hidden_dim, cfg.feature_dim, cfg.dropout, cfg.leakey)
        self.gc4_logstd = GCN(cfg.hidden_dim, cfg.feature_dim, cfg.dropout, cfg.leakey)
        self.user_union_mean = nn.Linear(cfg.feature_dim + cfg.feature_dim, cfg.feature_dim)
        self.user_union_logstd = nn.Linear(cfg.feature_dim + cfg.feature_dim, cfg.feature_dim)
        self.item_union_mean = nn.Linear(cfg.feature_dim + cfg.feature_dim, cfg.feature_dim)
        self.item_union_logstd = nn.Linear(cfg.feature_dim + cfg.feature_dim, cfg.feature_dim)

    @staticmethod
    def _kld_gauss(mu_1, logsigma_1, mu_2, logsigma_2):
        """KL(N(mu_1, sigma_1) || N(mu_2, sigma_2)), summed over dim 0 after mean over batch."""
        sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_1))
        sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(logsigma_2))
        q_target = Normal(mu_1, sigma_1)
        q_context = Normal(mu_2, sigma_2)
        return kl_divergence(q_target, q_context).mean(dim=0).sum()

    def reparameters(self, mean, logstd):
        sigma = torch.exp(0.1 + 0.9 * F.softplus(logstd))
        gaussian_noise = torch.randn(mean.size(0), self.cfg.hidden_dim, device=mean.device)
        if self.gc1.training:
            sampled_z = gaussian_noise * torch.exp(sigma) + mean
        else:
            sampled_z = mean
        kld_loss = self._kld_gauss(mean, logstd, torch.zeros_like(mean), torch.ones_like(logstd))
        return sampled_z, kld_loss

    def forward(self, ufea, vfea, UV_adj, VU_adj):
        user, user_kld = self.forward_user(ufea, vfea, UV_adj, VU_adj)
        item, item_kld = self.forward_item(ufea, vfea, UV_adj, VU_adj)
        self.kld_loss = user_kld + item_kld
        return user, item

    def forward_user(self, ufea, vfea, UV_adj, VU_adj):
        User_ho = self.gc1(ufea, VU_adj)
        User_ho_mean = self.gc3_mean(User_ho, UV_adj)
        User_ho_logstd = self.gc3_logstd(User_ho, UV_adj)
        User_ho_mean = self.user_union_mean(torch.cat((User_ho_mean, ufea), dim=1))
        User_ho_logstd = self.user_union_logstd(torch.cat((User_ho_logstd, ufea), dim=1))
        return self.reparameters(User_ho_mean, User_ho_logstd)

    def forward_item(self, ufea, vfea, UV_adj, VU_adj):
        Item_ho = self.gc2(vfea, UV_adj)
        Item_ho_mean = self.gc4_mean(Item_ho, VU_adj)
        Item_ho_logstd = self.gc4_logstd(Item_ho, VU_adj)
        Item_ho_mean = self.item_union_mean(torch.cat((Item_ho_mean, vfea), dim=1))
        Item_ho_logstd = self.item_union_logstd(torch.cat((Item_ho_logstd, vfea), dim=1))
        return self.reparameters(Item_ho_mean, Item_ho_logstd)

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        """Share branch: only the user side, returns (mean, logstd) WITHOUT reparam.
        The top-level model reparameterizes the fused cross-domain distribution once."""
        User_ho = self.gc1(ufea, VU_adj)
        User_ho_mean = self.gc3_mean(User_ho, UV_adj)
        User_ho_logstd = self.gc3_logstd(User_ho, UV_adj)
        User_ho_mean = self.user_union_mean(torch.cat((User_ho_mean, ufea), dim=1))
        User_ho_logstd = self.user_union_logstd(torch.cat((User_ho_logstd, ufea), dim=1))
        return User_ho_mean, User_ho_logstd


class singleVBGE(nn.Module):
    """Single-domain disentangled GNN-VAE.

    ``forward`` returns (user, item) sampled representations and leaves
    ``self.kld_loss`` on the last encoder layer (accessed by the top model).
    ``forward_user_share`` returns the per-domain (mean, logstd) of the shared
    user distribution, used to align against the cross-domain fused distribution.
    """

    def __init__(self, cfg):
        super(singleVBGE, self).__init__()
        self.cfg = cfg
        self.layer_number = cfg.n_layers
        self.encoder = nn.ModuleList(
            [_DGCNLayer(cfg) for _ in range(cfg.n_layers - 1)] + [_LastLayer(cfg)]
        )
        self.dropout = cfg.dropout

    def forward(self, ufea, vfea, UV_adj, VU_adj):
        learn_user = ufea
        learn_item = vfea
        for layer in self.encoder:
            learn_user = F.dropout(learn_user, self.dropout, training=self.training)
            learn_item = F.dropout(learn_item, self.dropout, training=self.training)
            learn_user, learn_item = layer(learn_user, learn_item, UV_adj, VU_adj)
        return learn_user, learn_item

    def forward_user_share(self, ufea, UV_adj, VU_adj):
        learn_user = ufea
        for layer in self.encoder[:-1]:
            learn_user = F.dropout(learn_user, self.dropout, training=self.training)
            learn_user = layer.forward_user_share(learn_user, UV_adj, VU_adj)
        mean, sigma = self.encoder[-1].forward_user_share(learn_user, UV_adj, VU_adj)
        return mean, sigma


# ----------------------------------------------------------------------
# Cross-domain shared DGCN-VAE
# ----------------------------------------------------------------------
class _CrossDGCNLayer(nn.Module):
    """Cross-domain hidden layer: propagate each domain's user on its own graph,
    then fuse the two domains' user features with a fixed ``rate`` weight."""

    def __init__(self, cfg):
        super(_CrossDGCNLayer, self).__init__()
        self.cfg = cfg
        self.gc1 = GCN(cfg.feature_dim, cfg.hidden_dim, cfg.dropout, cfg.leakey)
        self.gc2 = GCN(cfg.feature_dim, cfg.hidden_dim, cfg.dropout, cfg.leakey)
        self.gc3 = GCN(cfg.hidden_dim, cfg.feature_dim, cfg.dropout, cfg.leakey)
        self.gc4 = GCN(cfg.hidden_dim, cfg.feature_dim, cfg.dropout, cfg.leakey)
        self.source_user_union = nn.Linear(cfg.feature_dim + cfg.feature_dim, cfg.feature_dim)
        self.target_user_union = nn.Linear(cfg.feature_dim + cfg.feature_dim, cfg.feature_dim)
        self.register_buffer('source_rate', torch.tensor(cfg.rate).view(-1))

    def forward(self, source_ufea, target_ufea, source_UV_adj, source_VU_adj,
                target_UV_adj, target_VU_adj):
        source_User_ho = self.gc3(self.gc1(source_ufea, source_VU_adj), source_UV_adj)
        target_User_ho = self.gc4(self.gc2(target_ufea, target_VU_adj), target_UV_adj)
        source_User = self.source_user_union(torch.cat((source_User_ho, source_ufea), dim=1))
        target_User = self.target_user_union(torch.cat((target_User_ho, target_ufea), dim=1))
        # Both outputs are the same rate-weighted blend (matches upstream behavior);
        # downstream treats them as a single fused user representation.
        fused = self.source_rate * F.relu(source_User) + \
                (1 - self.source_rate) * F.relu(target_User)
        return fused, fused


class _CrossLastLayer(nn.Module):
    """Cross-domain final layer producing fused (mean, logstd) for the shared user."""

    def __init__(self, cfg):
        super(_CrossLastLayer, self).__init__()
        self.cfg = cfg
        self.gc1 = GCN(cfg.feature_dim, cfg.hidden_dim, cfg.dropout, cfg.leakey)
        self.gc2 = GCN(cfg.feature_dim, cfg.hidden_dim, cfg.dropout, cfg.leakey)
        self.gc3_mean = GCN(cfg.hidden_dim, cfg.feature_dim, cfg.dropout, cfg.leakey)
        self.gc3_logstd = GCN(cfg.hidden_dim, cfg.feature_dim, cfg.dropout, cfg.leakey)
        self.gc4_mean = GCN(cfg.hidden_dim, cfg.feature_dim, cfg.dropout, cfg.leakey)
        self.gc4_logstd = GCN(cfg.hidden_dim, cfg.feature_dim, cfg.dropout, cfg.leakey)
        self.source_user_union_mean = nn.Linear(cfg.feature_dim + cfg.feature_dim, cfg.feature_dim)
        self.source_user_union_logstd = nn.Linear(cfg.feature_dim + cfg.feature_dim, cfg.feature_dim)
        self.target_user_union_mean = nn.Linear(cfg.feature_dim + cfg.feature_dim, cfg.feature_dim)
        self.target_user_union_logstd = nn.Linear(cfg.feature_dim + cfg.feature_dim, cfg.feature_dim)
        self.register_buffer('source_rate', torch.tensor(cfg.rate).view(-1))

    def forward(self, source_ufea, target_ufea, source_UV_adj, source_VU_adj,
                target_UV_adj, target_VU_adj):
        source_User_ho = self.gc1(source_ufea, source_VU_adj)
        source_User_ho_mean = self.gc3_mean(source_User_ho, source_UV_adj)
        source_User_ho_logstd = self.gc3_logstd(source_User_ho, source_UV_adj)

        target_User_ho = self.gc2(target_ufea, target_VU_adj)
        target_User_ho_mean = self.gc4_mean(target_User_ho, target_UV_adj)
        target_User_ho_logstd = self.gc4_logstd(target_User_ho, target_UV_adj)

        source_User_mean = self.source_user_union_mean(
            torch.cat((source_User_ho_mean, source_ufea), dim=1))
        source_User_logstd = self.source_user_union_logstd(
            torch.cat((source_User_ho_logstd, source_ufea), dim=1))

        target_User_mean = self.target_user_union_mean(
            torch.cat((target_User_ho_mean, target_ufea), dim=1))
        target_User_logstd = self.target_user_union_logstd(
            torch.cat((target_User_ho_logstd, target_ufea), dim=1))

        mean = self.source_rate * source_User_mean + (1 - self.source_rate) * target_User_mean
        logstd = self.source_rate * source_User_logstd + (1 - self.source_rate) * target_User_logstd
        return mean, logstd


class crossVBGE(nn.Module):
    """Cross-domain shared user distribution. Fuses source & target user features
    via ``rate`` and produces a single (mean, logstd) for the shared user latent."""

    def __init__(self, cfg):
        super(crossVBGE, self).__init__()
        self.cfg = cfg
        self.layer_number = cfg.n_layers
        self.encoder = nn.ModuleList(
            [_CrossDGCNLayer(cfg) for _ in range(cfg.n_layers - 1)] + [_CrossLastLayer(cfg)]
        )
        self.dropout = cfg.dropout

    def forward(self, source_ufea, target_ufea, source_UV_adj, source_VU_adj,
                target_UV_adj, target_VU_adj):
        learn_user_source = source_ufea
        learn_user_target = target_ufea
        for layer in self.encoder[:-1]:
            learn_user_source = F.dropout(learn_user_source, self.dropout, training=self.training)
            learn_user_target = F.dropout(learn_user_target, self.dropout, training=self.training)
            learn_user_source, learn_user_target = layer(
                learn_user_source, learn_user_target,
                source_UV_adj, source_VU_adj, target_UV_adj, target_VU_adj)
        mean, sigma = self.encoder[-1](
            learn_user_source, learn_user_target,
            source_UV_adj, source_VU_adj, target_UV_adj, target_VU_adj)
        return mean, sigma
