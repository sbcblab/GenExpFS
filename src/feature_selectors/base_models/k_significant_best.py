import numpy as np


from .base_selector import BaseSelector, ResultType


class KSignificantBestFeatureSelector(BaseSelector):

    result_type = ResultType.WEIGHTS

    def __init__(self, score_func, n_features=None, p_threshold=0.05):
        super().__init__(n_features)
        self.score_func = score_func
        self.p_threshold = p_threshold
        self._weights, self._p_values = None, None

    def fit(self, X, y, **kwargs):
        self.check_already_fitted()
        self._X = X
        n_samples, n_features = X.shape

        score_func_result = self.score_func(X, y, **kwargs)

        if isinstance(score_func_result, (list, tuple)):
            weights, self._p_values = score_func_result
            self._p_values = np.asarray(self._p_values)
        else:
            weights = score_func_result

        if self._p_values is not None:
            self._selected = np.argwhere(self._p_values < self.p_threshold)
        else:
            self._selected = np.arange(n_features)

        self._weights = np.asarray(weights)
        self._rank = np.argsort(self._weights)[::-1]

        if self._n_features is not None:
            self._selected = self._rank[:self._n_features]
            self._support_mask = np.zeros(X.shape[1])
            self._support_mask[self._selected] = True

        self._fitted = True
        return self
