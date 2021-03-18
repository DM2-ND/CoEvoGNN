"""
Load temporal graphs and temporal attributes
"""

import collections
import copy


def load_temporalgraphs(nodes_f, temporalgraphs_f):
    nodes = []
    ''' All unique nodes across times '''
    with open(nodes_f, 'r') as f:
        for l in f:
            ts = l.strip().split('\t')
            node = ts[0]
            nodes.append(node)
    assert len(nodes) == len(set(nodes))  # Check all nodes are unique
    _node2ni = {node: ni for ni, node in enumerate(nodes)}

    ''' Raw edgelist (un-indexed) for each time '''
    _time2ti = collections.OrderedDict()
    _time2raw_edgelist = collections.defaultdict(list)
    with open(temporalgraphs_f, 'r') as f:
        for l in f:
            ts = l.strip().split(',')
            time = ts[0]
            if time not in _time2ti:
                _time2ti[time] = len(_time2ti)
            n_u = ts[1]
            n_v = ts[2]
            assert n_u in _node2ni and n_v in _node2ni  # Check all nodes are known
            if len(ts) == 4:
                weight = int(ts[3])
                for _ in range(weight):
                    _time2raw_edgelist[time].append([n_u, n_v])
            else:
                _time2raw_edgelist[time].append([n_u, n_v])
    times = list(_time2ti.keys())

    ''' List of edge list per time '''
    edgelists = []
    for time in times:
        raw_edgelist = _time2raw_edgelist[time]
        edgelists.append([[_node2ni[n_u], _node2ni[n_v]] for [n_u, n_v] in raw_edgelist])

    ''' List of adjacent list per time '''
    adjlists = []
    for edgelist in edgelists:
        _adjlist = collections.defaultdict(list)
        for (n_u, n_v) in edgelist:
            _adjlist[n_u].append(n_v)
            _adjlist[n_v].append(n_u)
        adjlists.append([_adjlist[ni] for ni in range(len(nodes))])

    # Sanity check: unique nodes per time
    assert all([set([n for e in edgelists[ti] for n in e]) == set([n for adj in adjlists[ti] for n in adj])
                for ti in range(len(times))])
    # Sanity check: num. of missing nodes per time
    assert all([len(set(range(len(nodes))) - set([n for e in edgelists[ti] for n in e])) ==
                sum([1 if len(adj) == 0 else 0 for adj in adjlists[ti]])
                for ti in range(len(times))])
    # Sanity check: num. of edges per time
    assert all([len(edgelists[ti])*2 == sum([len(adj) for adj in adjlists[ti]]) for ti in range(len(times))])

    return nodes, times, edgelists, adjlists


def load_temporalfeatures(features_f, temporalfeatures_f, nodes, times):
    _node2ni = {node: ni for ni, node in enumerate(nodes)}
    _time2ti = {time: ti for ti, time in enumerate(times)}

    ''' All unique features across time '''
    features = []
    with open(features_f, 'r') as f:
        for l in f:
            ts = l.strip().split('\t')
            feat = ts[0]
            features.append(feat)
    assert len(features) == len(set(features))  # Check all features are unique
    _feat2fi = {feat: fi for fi, feat in enumerate(features)}

    ''' Raw mapping: time -> node -> list of features (un-indexed) '''
    _time2node2features = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(temporalfeatures_f, 'r') as f:
        for l in f:
            ts = l.strip().split(',')
            assert len(ts) == 4
            time = ts[0]
            node = ts[1]
            feat = ts[2]
            freq = int(ts[3])
            # Sanity check: time, node, and feature are all known
            assert time in _time2ti and node in _node2ni and feat in _feat2fi
            for _ in range(freq):
                _time2node2features[time][node].append(feat)
    assert set(_time2node2features.keys()) == set(times)

    ''' List of node feature list per time '''
    featlists = []
    for time in times:
        featlists.append([[_feat2fi[feat] for feat in _time2node2features[time][node]] for node in nodes])

    return features, featlists


def concat_multi_features(multi_features, multi_featlists):
    features = ['{}f{}'.format(feat, mi) for mi, features in enumerate(multi_features) for feat in features]
    _feat_sizes = [len(features) for features in multi_features]
    _cum_feat_sizes = [sum(_feat_sizes[:i + 1]) for i in range(len(_feat_sizes))]

    # Sanity check: num. of times are the same
    assert len(set([len(featlists) for featlists in multi_featlists])) == 1
    _num_time = len(multi_featlists[0])
    # Sanity check: num. of nodes are the same
    assert len(set([len(featlist) for featlists in multi_featlists for featlist in featlists])) == 1
    _num_node = len(multi_featlists[0][0])

    featurelists = copy.deepcopy(multi_featlists[0])
    for mi in range(1, len(multi_featlists)):
        for ti in range(_num_time):
            for ni in range(_num_node):
                _remapped_feats = [feat + _cum_feat_sizes[mi - 1] for feat in multi_featlists[mi][ti][ni]]
                featurelists[ti][ni].extend(_remapped_feats)

    return features, featurelists
