import numpy as np
from numpy import int64


def ranked_to_permutation_list(a, size=None, fill=0):
    _size = size if size is not None else len(a)
    _a = np.full(_size, fill, dtype=int64)
    _a[a] = np.arange(len(a))
    return _a


def partial_rank(a, b):
    m = np.stack((a, b)).max()

    _a = ranked_to_permutation_list(a, m + 1, m + 1)
    _b = ranked_to_permutation_list(b, m + 1, m + 1)

    x, y = np.array([(x, y) for x, y in zip(_a, _b) if not (x == m and y == m)]).T
    return x, y


def rank_from_weights(weights, n=None):
    return np.argsort(weights)[::-1][:n]


def keep_top_k(weights, k, set_others_to=0):
    feat_rank = rank_from_weights(weights, k)
    new_weights = np.zeros(len(weights))
    new_weights[feat_rank] = np.array(weights)[feat_rank]
    return new_weights
