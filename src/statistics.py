from itertools import combinations

import numpy as np
from scipy.stats import kruskal
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from features import is_discrete


def mutual_information(X, y):
    mutual_info = mutual_info_classif if is_discrete(y) else mutual_info_regression

    if len(X.shape) == 1:
        X = np.array([X]).T

    return mutual_info(X, y, discrete_features=is_discrete(X))


def single_feature_kruskal_wallis(X, y):
    return kruskal(*[X[y == c] for c in np.unique(y)])


def kruskal_wallis(X, y):
    results = np.array([single_feature_kruskal_wallis(x, y) for x in X.T])
    scores, pvalues = results.T
    return scores, pvalues


def spearmans_rank(a, b):
    _a = np.array(a) + 1
    _b = np.array(b) + 1
    m = len(_a)
    return 1 - 6 * np.power(_a - _b, 2).sum() / (m * (np.power(m, 2) - 1))


def spearman_rank_ties(a, b):
    _a = np.array(a) + 1
    _b = np.array(b) + 1
    a_mean = _a.mean()
    b_mean = _b.mean()

    div = ((_a - a_mean) * (_b - b_mean)).sum()
    quo = np.sqrt(np.power(_a - a_mean, 2).sum() * np.power(_b - b_mean, 2).sum())

    return div / quo


def canberra_rank(a, b):
    _a = np.array(a) + 1
    _b = np.array(b) + 1
    return (np.abs(_a - _b) / (_a + _b)).sum()


def kendalls_rank(a, b):
    _a = np.array(a) + 1
    _b = np.array(b) + 1

    a_pairs = combinations(_a, 2)
    b_pairs = combinations(_b, 2)

    concordant = [x[0][0] < x[0][1] and x[1][0] < x[1][1] for x in zip(a_pairs, b_pairs)]

    return (2 * np.count_nonzero(concordant) - len(concordant)) / len(concordant)
