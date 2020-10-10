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
        self._support_mask = None
        self._selected = None
        self._rank = None
        self._weights = None
        self._fitted = False

    def fit(X, y):
        raise NotImplementedError()

    def _check_fit(self):
        if not self._fitted:
            raise Exception("Model is not fitted!")

    def get_features(self, k=None):
        self._check_fit()
        feats = self._X[:, self._support_mask]
        return feats[:, :k] if k else feats

    def get_selected(self):
        self._check_fit()
        return self._selected

    def get_rank(self, k):
        self._check_fit()
        if self._rank:
            return self._rank[:k] if k else self._rank
        raise Exception("This selector does not return feature ranks!")

    def get_weights(self, k):
        self._check_fit()
        if self._weights:
            return self._weights[:k] if k else self._weights
        raise Exception("This selector does not return feature weights!")

    def get_mask(self):
        self._check_fit()
        return self._support_mask

    def get_support(self, indices=False):
        self._check_fit()
        return self._selected if indices else self._support_mask

    def transform(self, X):
        return X[:, self._support_mask]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
