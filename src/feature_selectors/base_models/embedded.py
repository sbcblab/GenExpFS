import numpy as np


from .base_selector import BaseSelector, ResultType


class BaseEmbeddedFeatureSelector(BaseSelector):
    def __init__(self, model, weights_attr, is_callable=False, n_features=None, result_type=ResultType.WEIGHTS):
        super().__init__(result_type, n_features)
        self._model = model
        self._weights_attr = weights_attr
        self._is_callable = is_callable

    def fit(self, X, y, **kwargs):
        self.check_already_fitted()
        self._X = X

        self._model.fit(X, y, **kwargs)

        weights = eval(f'self._model.{self._weights_attr}{"()" if self._is_callable else ""}')
        self._weights = np.copy(weights)
        self._rank = np.argsort(self._weights)[::-1]

        if self._n_features is not None:
            self._selected = self._rank[:self._n_features]
            self._support_mask = np.zeros(X.shape[1])
            self._support_mask[self._rank] = True

        self._fitted = True
        return self
