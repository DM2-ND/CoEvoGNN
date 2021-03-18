"""
Model of CoEvoGNN
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoEvoGNN(nn.Module):
    def __init__(self, H_0, adjlists, T, K, num_neig_samples=20):
        super(CoEvoGNN, self).__init__()
        ''' Hidden state of all nodes at t=0 '''
        assert len(set([len(h) for h in H_0])) == 1
        self.H_0 = H_0
        self.num_node = self.H_0.shape[0]
        self.hid_emb_dim = self.H_0.shape[1]
        ''' Neighbor functions t=0...T (adjacent lists) '''
        assert all([len(adjlist) == self.num_node for adjlist in adjlists])
        self.adjlists = adjlists
        ''' Hyper-parameters '''
        assert T >= 1
        self.T = T
        assert K <= self.T + 1  # assert K <= self.T
        self.K = K
        ''' Misc. '''
        self.num_neig_samples = num_neig_samples
        self._agg_emb_dim = self.hid_emb_dim
        ''' Weights: W^(k) '''
        self.weight_W_ks = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(self.hid_emb_dim + self._agg_emb_dim, self.hid_emb_dim))
             for _ in range(self.K)])
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.)

    def _step_k(self, H_ts, t, k, _re=False):
        assert k <= t
        self_feats = H_ts[t - k]
        assert torch.sum(self_feats) != 0
        agg_feats = torch.zeros_like(self_feats)
        ''' Neighborhood sampling '''
        _rc_f = np.random.choice
        _neiglist = [_rc_f(list(adjs), self.num_neig_samples, replace=False) if len(adjs) >= self.num_neig_samples
                     else (_rc_f(list(adjs), self.num_neig_samples, replace=True) if _re else adjs)
                     for adjs in self.adjlists[t - k]]
        ''' Aggregation: non-parametric mean '''
        _mask = torch.zeros(self.num_node, self.num_node)
        _r_indices = [n for n in range(len(_neiglist)) for _ in range(len(_neiglist[n]))]
        _c_indices = [neig_n for neigs in _neiglist for neig_n in neigs]
        _mask[_r_indices, _c_indices] = 1
        _num_neig = _mask.sum(1, keepdim=True)
        _mask_mean = _mask.div(_num_neig)
        agg_feats = torch.mm(_mask_mean, self_feats)
        assert agg_feats.shape == (self.num_node, self._agg_emb_dim)
        return torch.cat((self_feats, agg_feats), dim=1)

    def _step(self, H_ts, t):
        assert H_ts.shape == (t, self.num_node, self.hid_emb_dim)
        ''' Transform K previous H '''
        _H_t_ks = [torch.mm(self._step_k(H_ts, t, k), self.weight_W_ks[k - 1])
                   for k in range(1, min(t, self.K) + 1)]
        ''' Fuse K transformed previous H'''
        H_t = F.relu(torch.sum(torch.stack(_H_t_ks), 0))  # Nonlinearity outside Sigma
        ''' Row-wise L2 normalization '''
        H_t = F.normalize(H_t, p=2, dim=1)
        return H_t

    def forward(self, t_train, t_forecast=1):
        assert t_train <= self.T
        H_ts = torch.unsqueeze(self.H_0, 0)  # Hidden state of all nodes at t=0,1...t_train...(t_train+t_forecast)
        ''' Generate H_t cascade (train and forecast)'''
        for t in range(1, t_train + t_forecast + 1):
            H_ts = torch.cat((H_ts, torch.unsqueeze(self._step(H_ts, t), 0)))
        ''' Split H_ts_train and H_ts_test '''
        return H_ts[1:t_train + 1], H_ts[t_train + 1:]


class AttributeInference(nn.Module):
    def __init__(self, hid_emb_dim, raw_emb_dim):
        super(AttributeInference, self).__init__()
        self.hid_emb_dim = hid_emb_dim
        self.raw_emb_dim = raw_emb_dim
        ''' Transformation from H space to X space '''
        self.weight_M = nn.Parameter(torch.FloatTensor(self.hid_emb_dim, self.raw_emb_dim))
        ''' Sum squared errors of X - X^hat '''
        self.loss_mse = nn.MSELoss(reduction='sum')
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def infer(self, H_ts_forecast):
        assert len(H_ts_forecast.shape) == 3, H_ts_forecast.shape  # (num_forecast, num_node, hid_emb_dim)
        ''' Batch transform H_ts_forecast '''
        return torch.matmul(H_ts_forecast, self.weight_M)

    def loss_balance(self, H_ts_train, X_ts_train, X_ts_train_subsample_mask):
        assert H_ts_train.shape[:2] == X_ts_train.shape[:2]
        assert H_ts_train.shape[2] == self.hid_emb_dim
        assert X_ts_train.shape[2] == self.raw_emb_dim
        assert X_ts_train.shape == X_ts_train_subsample_mask.shape
        ''' Subsample majority of 0s in X_ts_train^hat and take mean of squared errors '''
        loss = self.loss_mse(torch.matmul(H_ts_train, self.weight_M) * X_ts_train_subsample_mask.float(), X_ts_train)
        _num_subsample = torch.sum(X_ts_train_subsample_mask).item()
        loss = loss / _num_subsample
        return loss


class LinkPrediction(nn.Module):
    def __init__(self, adjlists):
        super(LinkPrediction, self).__init__()
        assert len(set([len(adjlist) for adjlist in adjlists])) == 1
        self.adjlists = adjlists
        self.num_node = len(self.adjlists[0])
        self.nodelists = [{n for adj in adjlist for n in adj} for adjlist in self.adjlists]
        self.allnodes = set(list(range(self.num_node)))

    def loss_rw(self, node_vs, H_ts_train, L=2, R=20, Q=200, _in=True, _re=False):
        assert len(node_vs) == len(set(node_vs)) and max(node_vs) <= self.num_node
        loss = torch.zeros(len(H_ts_train))
        for t in range(1, len(H_ts_train) + 1):
            _loss_t = torch.zeros(len(node_vs))  # loss_pos + loss_neg for each node_v at time t
            for i, node_v in enumerate(node_vs):
                rws = self._basic_rw(node_v, t, L, R, Q, _in, _re)
                if rws is not None:  # If node_v appears at time t
                    assert rws.numel() > 0 and rws.shape[0] > 0 and rws.shape[1] == Q+1
                    pos_node_us, neg_node_us = rws[:, 0], rws[:, 1:]
                    ''' Loss of pos node pairs '''
                    assert len(pos_node_us.shape) == 1
                    loss_pos = torch.mv(H_ts_train[t - 1][pos_node_us], H_ts_train[t - 1][node_v])
                    ''' Loss of neg node pairs '''
                    neg_node_us = neg_node_us.contiguous().view(-1)
                    assert len(neg_node_us.shape) == 1
                    loss_neg = torch.mv(H_ts_train[t - 1][neg_node_us], H_ts_train[t - 1][node_v])
                    ''' Merge loss of pos, neg pairs '''
                    loss_pos = - torch.log(torch.sigmoid(loss_pos))
                    loss_pos = torch.mean(loss_pos)
                    loss_neg = - torch.log(torch.sigmoid(- loss_neg))
                    loss_neg = torch.tensor(Q) * torch.mean(loss_neg)
                    # Sum mean pos, mean neg loss of node n at time t
                    _loss_t[i] = torch.sum(torch.stack((loss_pos, loss_neg)))
            loss[t - 1] = torch.mean(_loss_t)
        return torch.mean(loss)

    def _basic_rw(self, node_v, t, L, R, Q, _in=True, _re=False):
        rws = None  # R by Q+1: 1st node of each row is pos node_u; the rest are neg node_u nodes
        if node_v in self.nodelists[t]:  # If node_v appears at time t
            ''' Build auxiliary list of 0,1,...,L hop neighbor nodes (at time t) '''
            _neigs_l = [{node_v}]
            _reached_nodes = set()
            _reached_nodes.update(_neigs_l[-1])
            for _ in range(L):
                _next_neigs = {next_n for prev_n in _neigs_l[-1]
                               for next_n in self.adjlists[t][prev_n] if next_n not in _reached_nodes}
                _neigs_l.append(_next_neigs)
                _reached_nodes.update(_next_neigs)
            ''' Pool for pos node_u nodes '''
            _neigs = _neigs_l[L] if not _in else _reached_nodes
            _neigs.discard(node_v)  # Remove self
            if len(_neigs) > 0:  # If pool is not empty (after removing self)
                ''' Sample R pos node_u '''
                if len(_neigs) >= R:
                    pos_node_us = torch.as_tensor(np.random.choice(list(_neigs), size=(R, 1), replace=False))
                else:
                    if _re:
                        pos_node_us = torch.as_tensor(np.random.choice(list(_neigs), size=(R, 1), replace=True))
                    else:
                        pos_node_us = torch.as_tensor(list(_neigs), dtype=torch.long).view(-1, 1)
                assert len(pos_node_us.shape) == 2 and pos_node_us.shape[0] <= R and pos_node_us.shape[1] == 1
                ''' Pool for neg node_u nodes '''
                _ex_neigs = self.nodelists[t] - _neigs
                _ex_neigs.discard(node_v)  # Remove self
                ''' For each one of R pos node_u, sample Q neg node_us '''
                neg_node_us = torch.as_tensor(np.random.choice(list(_ex_neigs),
                                                               size=(pos_node_us.shape[0], Q), replace=True))
                assert len(neg_node_us.shape) == 2
                assert neg_node_us.shape[0] == pos_node_us.shape[0] and neg_node_us.shape[1] == Q
                rws = torch.cat((pos_node_us, neg_node_us), 1)
                assert rws.numel() > 0 and rws.shape[0] > 0 and rws.shape[1] == Q+1
        return rws
