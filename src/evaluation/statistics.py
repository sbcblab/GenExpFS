from itertools import combinations

import numpy as np
from scipy.stats import kruskal
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from util.features import is_discrete
from .util import partial_rank, ranked_to_permutation_list


def mutual_information(X, y):
    mutual_info = mutual_info_classif if is_discrete(y) else mutual_info_regression

    if len(X.shape) == 1:
        X = np.array([X]).T

    return mutual_info(X, y, discrete_features=is_discrete(X))


def _single_feature_kruskal_wallis(X, y):
    return kruskal(*[X[y == c] for c in np.unique(y)])


def kruskal_wallis(X, y):
    if len(X.shape) == 1:
        return _single_feature_kruskal_wallis(X, y)

    results = np.array([_single_feature_kruskal_wallis(x, y) for x in X.T])
    scores, pvalues = results.T
    return scores, pvalues


def no_ties_spearmans_correlation(a, b):
    _a = np.array(a) + 1
    _b = np.array(b) + 1
    m = len(_a)
    return 1 - 6 * np.power(_a - _b, 2).sum() / (m * (np.power(m, 2) - 1))


def spearmans_correlation(a, b):
    return pearsons_correlation(a, b)


def spearmans_correlation_ranked_list(a, b):
    _a = ranked_to_permutation_list(a)
    _b = ranked_to_permutation_list(b)
    return spearmans_correlation(_a, _b)


def spearmans_correlation_partial_ranked_list(a, b):
    _a, _b = partial_rank(a, b)
    return spearmans_correlation(_a, _b)


def pearsons_correlation(a, b):
    _a = np.array(a) + 1
    _b = np.array(b) + 1
    a_mean = _a.mean()
    b_mean = _b.mean()

    div = ((_a - a_mean) * (_b - b_mean)).sum()
    quo = np.sqrt(np.power(_a - a_mean, 2).sum() * np.power(_b - b_mean, 2).sum())

    return div / quo


def pearsons_correlation_no_zeros(a, b):
    a, b = np.array([(a, b) for a, b in zip(a, b) if not (a == 0 and b == 0)]).T
    return pearsons_correlation(a, b)


def canberra_distance(a, b):
    if a == b:
        return 0

    _a, _b = np.array([(a, b) for a, b in zip(a, b) if not (a == b)]).T
    _a = np.array(_a)
    _b = np.array(_b)
    return (np.abs(_a - _b) / (np.abs(_a) + np.abs(_b))).sum()


def canberra_distance_ranked_list(a, b):
    _a = ranked_to_permutation_list(a)
    _b = ranked_to_permutation_list(b)
    return canberra_distance(_a, _b)


def canberra_distance_partial_ranked_list(a, b):
    _a, _b = partial_rank(a, b)
    return canberra_distance(_a, _b)


def kendalls_tau_coefficient(a, b):
    _a = np.array(a) + 1
    _b = np.array(b) + 1

    a_pairs = combinations(_a, 2)
    b_pairs = combinations(_b, 2)

    concordant = [x[0][0] < x[0][1] and x[1][0] < x[1][1] for x in zip(a_pairs, b_pairs)]

    return (2 * np.count_nonzero(concordant) - len(concordant)) / len(concordant)


def kendalls_tau_ranked_list(a, b):
    _a = ranked_to_permutation_list(a)
    _b = ranked_to_permutation_list(b)
    return kendalls_tau_coefficient(_a, _b)


def kendalls_tau_partial_ranked_list(a, b):
    _a, _b = partial_rank(a, b)
    return kendalls_tau_coefficient(_a, _b)
