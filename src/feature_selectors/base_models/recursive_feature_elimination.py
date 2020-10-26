import numpy as np
from sklearn.preprocessing import label_binarize

from .base_selector import BaseSelector, ResultType


class RFE(BaseSelector):

    result_type = ResultType.SUBSET

    def __init__(self, model, weights_attr, is_callable=False, n_features=None, encode_classes=False, verbose=0):
        super().__init__(n_features)
        self._model = model
        self._weights_attr = weights_attr
        self._is_callable = is_callable
        self._encode = encode_classes
        self._verbose = verbose

        if n_features is None:
            self.result_type = ResultType.RANK

    def _select_worst(self, X, y, remaining, **kwargs):
        if len(remaining) == 1:
            return remaining[0]

        _y = label_binarize(y, np.unique(y)) if self._encode else y
        self._model.fit(X[:, remaining], _y, **kwargs)

        weights = eval(f'self._model.{self._weights_attr}{"()" if self._is_callable else ""}')
        if len(weights.shape) > 1:
            weights = weights.sum(axis=0)

        worst_idx = np.argmin(weights)
        worst_feat = remaining[worst_idx]

        return worst_feat

    def fit(self, X, y, **kwargs):
        self.check_already_fitted()
        self._X = X

        num_feats = X.shape[1]

        n_features = 0 if self._n_features is None else self._n_features

        num_feats_to_remove = num_feats - np.min(np.array([num_feats, n_features]))

        remaining = list(range(num_feats))
        reverse_rank = []

        for i in range(num_feats_to_remove):
            selected_idx = self._select_worst(X, y, remaining, **kwargs)
            remaining.remove(selected_idx)
            reverse_rank.append(selected_idx)

            if self._verbose:
                print(f'[{i+1}/{num_feats_to_remove}] removed variable {selected_idx}.')

        if self._n_features is None:
            self._rank = reverse_rank[::-1]
        else:
            self._selected = np.copy(remaining)
            self._support_mask = np.zeros(X.shape[1])
            self._support_mask[self._rank] = True

        self._fitted = True
        return self
