"""
Train EvoGNN model
"""

from data_loader import *
from coevognn import *
from config import *
from utils import *

import argparse
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['2k', '10k'], required=False, default='2k',
                        help='Specify the dataset to use')
    parser.add_argument('--t_0', type=int, required=False, default=0,
                        help='Start index of available time points')
    parser.add_argument('--T', type=int, required=False, default=8,
                        help='Length of available training time points')
    parser.add_argument('--t_train', type=int, required=False, default=8,
                        help='Length of training time points (from --t_0)')
    parser.add_argument('--t_forecast', type=int, required=False, default=1,
                        help='Number of forecasting snapshots')
    parser.add_argument('--K', type=int, required=False, default=2,
                        help='Num of layers fusing new time point')
    parser.add_argument('--epochs', type=int, required=False, default=10,
                        help='Number of epochs for training')
    parser.add_argument('--H_0_npf', required=False, default=H_0_2k_npf,
                        help='File for initializing H_0')
    _args = parser.parse_args()
    return _args


if __name__ == '__main__':
    args = parse_args()
    """
    Load in files (dataset 2k)
    """
    print('===' * 30)
    print('Loading...')
    nodes_f, temporalgraphs_f, venues_f, temporalfeatures_venue_f, words_f, temporalfeatures_word_f = \
        con_dataset(args.dataset)
    print('Temporalgraphs file: {}'.format(temporalgraphs_f))
    print('Temporalfeatures file (venues): {}'.format(temporalfeatures_venue_f))
    print('Temporalfeatures file (words): {}'.format(temporalfeatures_word_f))
    nodes, times, edgelists, adjlists = load_temporalgraphs(nodes_f, temporalgraphs_f)
    _venues, _venuelists = load_temporalfeatures(venues_f, temporalfeatures_venue_f, nodes, times)
    _words, _wordlists = load_temporalfeatures(words_f, temporalfeatures_word_f, nodes, times)
    features, featurelists = concat_multi_features([_venues, _words], [_venuelists, _wordlists])
    edgesets, bi_edgesets, u_adjlists = summarize_temporalgraphs(nodes, times, edgelists, adjlists,
                                                                 _venues, _venuelists, _words, _wordlists,
                                                                 verbose=True)

    """
    Pre-processing
    """
    print('Pre-processing...')
    X_ts_np = build_X_ts_np(times, nodes, features, featurelists)
    X_0_np = X_ts_np[args.t_0]  # X_ts_np[0]
    _X_0_np_nodes = set([n for n, n_x in enumerate(X_0_np) if np.sum(n_x) > 0])
    ''' H_0 Initialization '''
    H_0_np = np.load(args.H_0_npf)
    _H_0_np_nodes = set([n for n, n_h in enumerate(H_0_np) if np.linalg.norm(n_h) > 0])
    assert _H_0_np_nodes.issubset(_X_0_np_nodes)
    # assert len(_X_0_np_nodes - _H_0_np_nodes) / len(_X_0_np_nodes) < 0.01  # Sparsity check
    X_ts = torch.as_tensor(X_ts_np, dtype=torch.float)
    H_0 = torch.as_tensor(H_0_np, dtype=torch.float)

    X_ts_train = X_ts[args.t_0 + 1: args.t_0 + args.t_train + 1]
    X_ts_test = X_ts[args.t_0 + args.t_train + 1: args.t_0 + args.t_train + args.t_forecast + 1]
    edgesets_train = bi_edgesets[args.t_0 + 1: args.t_0 + args.t_train + 1]
    edgesets_test = bi_edgesets[args.t_0 + args.t_train + 1: args.t_0 + args.t_train + args.t_forecast + 1]
    u_adjlists = u_adjlists[args.t_0:]  # Discard before t_0
    raw_emb_dim = X_ts.shape[2]
    hid_emb_dim = H_0.shape[1]

    """
    Initializations
    """
    print('===' * 30)
    print('Arguments:')
    _options = vars(args)
    for _op, _op_v in _options.items():
        if _op == 't_0':
            print(' - t_0: ({})'.format(times[args.t_0]))
        elif _op == 'T':
            print(' - T: {} ({})'.format(args.T, ' '.join(times[args.t_0 + 1: args.t_0 + args.T + 1])))
        elif _op == 't_train':
            print(' - t_train: {} ({})'.format(args.t_train,
                                               ' '.join(times[args.t_0 + 1: args.t_0 + args.t_train + 1])))
        elif _op == 't_forecast':
            print(' - t_forecast: ({})'
                  .format(' '.join(times[args.t_0 + args.t_train + 1: args.t_0 + args.t_train + args.t_forecast + 1])))
        else:
            print(' - {}: {}'.format(_op, _op_v))

    """
    Training
    """
    coevognn = CoEvoGNN(H_0=H_0, adjlists=u_adjlists, T=args.T, K=args.K)
    attr_infer = AttributeInference(hid_emb_dim=hid_emb_dim, raw_emb_dim=raw_emb_dim)
    link_pred = LinkPrediction(adjlists=u_adjlists)  # Non-parametric model
    models = [coevognn, attr_infer, link_pred]
    params = [param for model in models for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.025, weight_decay=1e-6)

    _ep_losses, _ep_metrics, _ep_times = [], [], []  # Training log
    _optimal_ep, _optimal_metrics_score = 0, 0
    print('===' * 30)
    print('Training CoEvoGNN ({:,})...'.format(args.epochs), flush=True)
    for ep in range(args.epochs):
        s_t = time.time()

        ''' Initializations '''
        optimizer.zero_grad()
        X_ts_train_subsample_mask = gen_subsample_mask(X_ts_train)
        X_ts_test_subsample_mask = gen_subsample_mask(X_ts_test)

        ''' Forward '''
        H_ts_train, H_ts_forecast = coevognn(args.t_train, args.t_forecast)
        loss_attr = 100 * attr_infer.loss_balance(H_ts_train, X_ts_train, X_ts_train_subsample_mask)
        loss_stru = 1 * link_pred.loss_rw(np.random.choice(list(range(len(nodes))), len(nodes) // 10,
                                                           replace=False), H_ts_train)
        loss = loss_attr + loss_stru
        _ep_losses.append([loss, loss_attr, loss_stru])

        ''' Test '''
        for param in params:
            param.requires_grad = False
        X_ts_forecast = attr_infer.infer(H_ts_forecast)
        # Attribute metrics
        maes, rmses = eval_attr(X_ts_forecast, X_ts_test, X_ts_test_subsample_mask)
        # Structure metrics
        stru_metrics = eval_stru_basic(H_ts_forecast, edgesets_test)
        _ep_metrics.append({'maes': maes, 'rmses': rmses, 'pr_aucs': stru_metrics['skl_pr_aucs'],
                            'f1_maxs': stru_metrics['skl_f1_maxs'], 'f1_ths': stru_metrics['skl_f1_ths']})
        for param in params:
            param.requires_grad = True

        ''' Backpropagation '''
        loss.backward()
        for model in models:
            nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        optimizer.zero_grad()

        ''' Epoch summary '''
        e_t = time.time()
        _ep_time = e_t - s_t
        _ep_times.append(_ep_time)
        print('Ep.{:03d} Loss: {:6.3f} = {:6.3f} + {:6.3f} ({:5.1f} secs)'
              .format(ep + 1, loss, loss_attr, loss_stru, _ep_time), flush=True)
        for t in range(args.t_forecast):
            # Attribute metrics
            print('- [{}] MAE: {:5.4f}, RMSE: {:5.4f};'
                  .format(times[args.t_0 + args.t_train + 1 + t], maes[t], rmses[t]), end=' ')
            # Structure metrics
            print('PR_AUC: {:5.4f} ({:5.4f}), F1: {:5.4f} @ p={:4.3f} ({:5.4f} @ p={:4.3f}); P@ks: {}'
                  .format(stru_metrics['skl_pr_aucs'][t], stru_metrics['man_pr_aucs'][t],
                          stru_metrics['skl_f1_maxs'][t], stru_metrics['skl_f1_ths'][t],
                          stru_metrics['man_f1_maxs'][t], stru_metrics['man_f1_ths'][t],
                          ''.join(['{:,}:{:5.4f}, '.format(k, p) for (k, p) in stru_metrics['man_p_at_ks'][t]])))
        print('---' * 30, flush=True)

    ''' Training summary '''
    print('===' * 30)
    print('Done')
    print('Training time: total: {:5.1f} secs; avg. epoch time: {:5.1f} secs;'
          .format(np.sum(_ep_times), np.mean(_ep_times)))
    print('===' * 30)
