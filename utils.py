"""
Utils
"""

import itertools
import collections

import numpy as np
from sklearn.metrics import *

import torch
import torch.nn as nn


def summarize_temporalgraphs(nodes, times, edgelists, adjlists, _venues, _venuelists, _words, _wordlists, verbose=True):
    _nodelists = [{n for adj in adjlist for n in adj} for adjlist in adjlists]  # Node lists for each t
    ''' Edge sets (original and bi-direction) for each t '''
    edgesets = [{(n_v, n_u) for (n_v, n_u) in edgelist} for edgelist in edgelists]
    bi_edgesets = [{(n_v, n_u) for (n_v, n_u) in edgelist} | {(n_u, n_v) for (n_v, n_u) in edgelist}
                   for edgelist in edgelists]
    ''' Adjacent lists of unique nodes w/ self connection for each t '''
    u_adjlists = [[{n, *adj} for n, adj in enumerate(adjlist)] for adjlist in adjlists]
    if verbose:
        print('Dataset:')
        print('---'*20)
        print('{:5s} | {:>9s} {:>9s} {:>9s} | {:>9s} {:>9s}'
              .format('TIME', '|V|', '|E|', 'Dens.(%)', '#V.', '#W.'))
        print('---' * 20)
        for t, time_name in enumerate(times):
            _num_node = len(_nodelists[t])
            _num_edge = len(edgelists[t])
            _den = 2 * _num_edge / _num_node / (_num_node - 1)  # Undirected graph
            _num_venue = len(set(v for venues in _venuelists[t] for v in venues))
            _num_word = len(set(w for words in _wordlists[t] for w in words))
            print('{:5s} | {:9,d} {:9,d} {:9.3%} | {:9,d} {:9,d}'
                  .format(time_name, _num_node, _num_edge, _den, _num_venue, _num_word))
        print('---' * 20)
        print('{} | {:9,d} {:9s} {:9s} | {:9,d} {:9,d}'
              .format('Total', len(nodes), '-'.center(9), '-'.center(9), len(_venues), len(_words)))
        print('---' * 20)
    return edgesets, bi_edgesets, u_adjlists


def merge_u_adjlists(u_adjlists):
    num_t = len(u_adjlists)
    num_node = len(u_adjlists[0])
    return [set.union(*[u_adjlists[t][n] for t in range(num_t)]) for n in range(num_node)]


def merge_edgesets(edgesets):
    num_t = len(edgesets)
    return set.union(*[edgesets[t] for t in range(num_t)])


def build_X_ts_np(times, nodes, features, featurelists):
    assert len(featurelists) == len(times)
    assert len(featurelists[0]) == len(nodes)
    assert max([f for featurelist in featurelists for features in featurelist for f in features]) <= len(features)
    ''' Build node attributes matrices '''
    X_ts_np = np.zeros((len(times), len(nodes), len(features)))
    for t in range(len(times)):
        _r_indices = [n for n in range(len(featurelists[t])) for _ in range(len(featurelists[t][n]))]
        _c_indices = [f for features in featurelists[t] for f in features]
        for _r, _c in zip(_r_indices, _c_indices):
            X_ts_np[t, _r, _c] += 1
    assert all([np.sum(X_ts_np[t]) == np.sum([len(features) for features in featurelists[t]])
                for t in range(len(times))])
    return X_ts_np


def gen_subsample_mask(X_ts, ratio=1):
    _X_ts_pos_mask = (X_ts > 0)
    _X_ts_density = torch.sum(_X_ts_pos_mask).item()/torch.numel(X_ts)
    _X_ts_zero_mask = (X_ts == 0)
    _subsample_mask = torch.rand_like(_X_ts_zero_mask, dtype=torch.float)
    _subsample_mask = _subsample_mask * _X_ts_zero_mask.float() <= (_X_ts_density * ratio)
    _subsample_mask = _subsample_mask + _X_ts_pos_mask
    return _subsample_mask


def eval_attr(X_ts_forecast, X_ts_test, X_ts_test_subsample_mask):
    assert X_ts_forecast.shape == X_ts_test.shape
    assert X_ts_forecast.shape == X_ts_test_subsample_mask.shape
    maes, rmses = [], []
    ''' Sum absolute/squared errors of X - X^hat '''
    loss_mae = nn.L1Loss(reduction='sum')
    loss_mse = nn.MSELoss(reduction='sum')
    ''' Subsample majority of 0s in X_ts_test^hat and take mean of absolute/squared errors '''
    t_test = X_ts_forecast.shape[0]
    for t in range(t_test):
        _num_subsample = torch.sum(X_ts_test_subsample_mask[t]).item()
        _mae = loss_mae(X_ts_forecast[t]*X_ts_test_subsample_mask[t].float(), X_ts_test[t]).item() / _num_subsample
        _mse = loss_mse(X_ts_forecast[t]*X_ts_test_subsample_mask[t].float(), X_ts_test[t]).item() / _num_subsample
        maes.append(_mae)
        rmses.append(np.sqrt(_mse))
    return maes, rmses


def eval_stru_basic(H_ts_forecast, edgesets_test, check=True, verbose=False):
    assert len(H_ts_forecast) == len(edgesets_test)
    H_ts_forecast = H_ts_forecast.detach().numpy()
    metrics = collections.defaultdict(list)
    t_test = H_ts_forecast.shape[0]
    num_node = H_ts_forecast.shape[1]
    for t in range(t_test):
        ''' Compute (score, label) for each node pair (w/o self connections) '''
        A_forecast = np.dot(H_ts_forecast[t], np.transpose(H_ts_forecast[t]))
        scores_labels = []
        for (n_u, n_v) in itertools.product(range(num_node), range(num_node)):
            if n_u != n_v:  # Exclude self connections
                scores_labels.append((A_forecast[n_u][n_v], 1 if (n_u, n_v) in edgesets_test[t] else 0))
        [scores, labels] = zip(*scores_labels)
        ''' Metrics: PR_AUC, F1_max, F1_threshold '''
        skl_pr_auc, skl_f1_max, skl_f1_th = _skl_metrics(scores, labels)
        metrics['skl_pr_aucs'].append(skl_pr_auc)
        metrics['skl_f1_maxs'].append(skl_f1_max)
        metrics['skl_f1_ths'].append(skl_f1_th)
        if check:
            _ks = [50, 100, 200]
            man_pr_auc, man_f1_max, man_f1_th, man_p_at_ks = _man_metrics(scores, labels, ks=_ks)
            metrics['man_pr_aucs'].append(man_pr_auc)
            metrics['man_f1_maxs'].append(man_f1_max)
            metrics['man_f1_ths'].append(man_f1_th)
            metrics['man_p_at_ks'].append(man_p_at_ks)
        if verbose:
            print('(eval_stru) edges pred/true/all: {:10,}/{:10,}/{:10,}'
                  .format(np.sum(scores >= skl_f1_th), np.sum(labels), len(scores)))
    return metrics


def _skl_metrics(scores, labels):
    assert len(scores) == len(labels)
    ''' PR_AUC '''
    skl_pr_auc = average_precision_score(y_true=labels, y_score=scores)
    ''' F1 '''
    _ps, _rs, _th = precision_recall_curve(y_true=labels, probas_pred=scores)
    skl_recs, skl_pres, skl_ths = _rs[::-1], _ps[::-1], _th[::-1]
    assert len(skl_recs) == len(skl_pres) and len(skl_recs) == len(skl_ths) + 1
    skl_f1s = 2 * skl_recs * skl_pres / (skl_recs + skl_pres + 1e-6)  # Numerical stability
    skl_f1_max = np.max(skl_f1s)
    skl_f1_th = skl_ths[np.argmax(skl_f1s) - 1]
    return skl_pr_auc, skl_f1_max, skl_f1_th


def _man_metrics(scores, labels, ks, interpolate=False):
    assert len(scores) == len(labels)
    ''' Sort by scores (descending) '''
    man_sls = sorted(zip(scores, labels), key=lambda x: (x[0]), reverse=True)
    [man_ss, man_ls] = zip(*man_sls)
    num_test, num_minority = len(man_ls), np.sum(man_ls)
    ''' Compute precision and recall at each point'''
    _man_ls_cum = np.add.accumulate(man_ls)
    man_rps = [(_man_ls_cum[i]/num_minority, _man_ls_cum[i]/(i+1)) for i in range(num_test)]
    [man_recs, man_pres] = zip(*man_rps)
    man_recs, man_pres = np.array(man_recs), np.array(man_pres)
    ''' F1 '''
    man_f1s = 2 * man_recs * man_pres / (man_recs + man_pres + 1e-6)  # Numerical stability
    man_f1_max = np.max(man_f1s)
    man_f1_th = man_ss[np.argmax(man_f1s)]
    ''' PR_AUC '''
    if interpolate:
        man_pres_interpolate = np.maximum.accumulate(man_pres[::-1])[::-1]
        man_pr_auc = auc(man_recs, man_pres_interpolate)
    else:
        man_pr_auc = auc(man_recs, man_pres)
    ''' Precision@ks'''
    man_p_at_ks = [(k, man_pres[k - 1]) for k in ks]
    return man_pr_auc, man_f1_max, man_f1_th, man_p_at_ks
