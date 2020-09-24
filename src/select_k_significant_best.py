import numpy as np


class SelectKSignificantBest:
    def __init__(self, score_func, k=20, p_threshold=0.05):
        self.score_func = score_func
        self.k = k
        self.p_threshold = p_threshold
        self.scores_, self.pvalues_ = None, None

    def fit(self, X, y, **kwargs):
        score_func_result = self.score_func(X, y, **kwargs)

        if isinstance(score_func_result, (list, tuple)):
            self.scores_, self.pvalues_ = score_func_result
            self.pvalues_ = np.asarray(self.pvalues_)
        else:
            self.scores_ = score_func_result
        self.scores_ = np.asarray(self.scores_)
        return self

    def _p_idx(self):
        return np.argwhere(self.pvalues_ < self.p_threshold)

    def _scores_idx(self):
        return np.argsort(self.scores_, kind="mergesort")

    def _best_idx(self):
        p_idx = self._p_idx()
        return [idx for idx in self._scores_idx() if idx in p_idx][-self.k:]

    def _get_support_mask(self):
        mask = np.zeros(self.scores_.shape, dtype=bool)
        mask[self._best_idx()] = 1
        return mask

    def get_suport(self, indices=False):
        mask = self._get_support_mask()
        return np.where(mask)[0] if indices else mask

    def transform(self, X):
        return X[:, self._get_support_mask()]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
