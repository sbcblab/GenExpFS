from enum import Enum


class SelectorKind(Enum):
    FILTER = "filter"
    WRAPPER = "wrapper"
    EMBEDDED = "embedded"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"


class BaseSelector:
    def __init__(self, kind: SelectorKind, n_features: int):
        self._kind = kind
        self._n_features = n_features

        self._X = None
        self._y = None
        self._mask = None
        self._selected = None
        self._rank = None
        self._weights = None
        self._fitted = False

    def fit(X, y):
        raise NotImplementedError()

    def _check_fit(self):
        if not self._fitted:
            raise Exception("Model is not fitted!")

    def get_features(self, k):
        self._check_fit()
        return self._X[:, self._mask]

    def get_selected(self, k):
        self._check_fit()
        return self.selected_

    def get_rank(self, k):
        self._check_fit()
        if self._rank:
            return self._rank
        raise Exception("This selector does not return feature ranks!")

    def get_weights(self, k):
        self._check_fit()
        if self._weights:
            return self._weights
        raise Exception("This selector does not return feature weights!")

    def get_mask(self):
        self._check_fit()
        return self._mask

    def get_support(self, indices=False):
        return self._selected if indices else self._mask

    def transform(self, X):
        return X[:, self._mask()]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
