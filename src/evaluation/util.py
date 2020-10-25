import numpy as np
from numpy.ctypeslib import int64


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
