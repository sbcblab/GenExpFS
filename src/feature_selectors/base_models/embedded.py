import numpy as np
from sklearn.preprocessing import label_binarize

from .base_selector import BaseSelector, ResultType


class BaseEmbeddedFeatureSelector(BaseSelector):

    result_type = ResultType.WEIGHTS

    def __init__(self, model, weights_attr, is_callable=False, n_features=None, encode_classes=False):
        super().__init__(n_features)
        self._model = model
        self._weights_attr = weights_attr
        self._is_callable = is_callable
        self._encode = encode_classes

    def fit(self, X, y, **kwargs):
        self.check_already_fitted()
        self._X = X

        _y = label_binarize(y, np.unique(y)) if self._encode else y

        self._model.fit(X, _y, **kwargs)

        weights = eval(f'self._model.{self._weights_attr}{"()" if self._is_callable else ""}')

        if len(weights.shape) > 1:
            weights = weights.sum(axis=0)

        self._weights = np.copy(weights)
        self._rank = np.argsort(self._weights)[::-1]

        if self._n_features is not None:
            self._selected = self._rank[:self._n_features]
            self._support_mask = np.zeros(X.shape[1])
            self._support_mask[self._rank] = True

        self._fitted = True
        return self
