import numpy as np
from scipy.stats import kruskal


def single_feature_kruskal_wallis(X, y):
    return kruskal(*[X[y == c] for c in np.unique(y)])


def kruskal_wallis(X, y):
    results = np.array([single_feature_kruskal_wallis(x, y) for x in X.T])
    scores, pvalues = results.T
    return scores, pvalues
