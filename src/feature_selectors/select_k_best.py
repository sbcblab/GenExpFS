import numpy as np
from sklearn.feature_selection import SelectKBest as SKSelectKBest


from .base_selector import BaseSelector, ResultType


class SelectKBest(BaseSelector):
    def __init__(self, score_func, n_features=None, p_threshold=0.05):
        super().__init__(ResultType.WEIGHTS, n_features)
        self._selector = SKSelectKBest(score_func, n_features, p_threshold)

    def fit(self, X, y, **kwargs):
        self.check_already_fitted()
        self._X = X

        self._selector.fit(X, y, **kwargs)

        self._weights = np.copy(self._selector.scores_)
        self._rank = np.argsort(self._weights)[::-1]
        self._fitted = True
        return self
