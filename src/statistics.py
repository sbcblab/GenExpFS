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
