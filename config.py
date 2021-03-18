"""
Global configurations
"""

import os

emb_dir = os.path.join('.', 'emb')


def con_dataset(d_name):
    if d_name == '2k':
        """ Co-author temporal networks 2k """
        nodes_f = os.path.join('data', 'node2author.n2k.csv')
        temporalgraphs_f = os.path.join('data', 'temporalgraph.n2k.csv')

        venues_f = os.path.join('data', 'venue2name.n2k.csv')
        temporalfeatures_venue_f = os.path.join('data', 'temporalfeature-venues.n2k.csv')

        words_f = os.path.join('data', 'word2name.n2k.csv')
        temporalfeatures_word_f = os.path.join('data', 'temporalfeature-words.n2k.csv')
    else:  # d_name == '10k'
        """ Co-author temporal networks 10k """
        nodes_f = os.path.join('data', 'node2author.n10k.csv')
        temporalgraphs_f = os.path.join('data', 'temporalgraph.n10k.csv')

        venues_f = os.path.join('data', 'venue2name.n10k.csv')
        temporalfeatures_venue_f = os.path.join('data', 'temporalfeature-venues.n10k.csv')

        words_f = os.path.join('data', 'word2name.n10k.csv')
        temporalfeatures_word_f = os.path.join('data', 'temporalfeature-words.n10k.csv')
    _fs = [nodes_f, temporalgraphs_f, venues_f, temporalfeatures_venue_f, words_f, temporalfeatures_word_f]
    assert all([os.path.exists(_f) for _f in _fs])
    return _fs


""" Cache files """
H_0_2k_npf = os.path.join(emb_dir, 'H_0.2k.npy')
H_0_10k_npf = os.path.join(emb_dir, 'H_0.10k.npy')
