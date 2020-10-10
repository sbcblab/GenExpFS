from abc import ABC, abstractmethod
from enum import Enum


class SelectorKind(Enum):
    FILTER = "filter"
    WRAPPER = "wrapper"
    EMBEDDED = "embedded"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"


class ResultType(Enum):
    COMPLETE_WEIGHTS = "complete_weights"
    WEIGHTS = "weights"
    COMPLETE_RANK = "complete_rank"
    RANK = "rank"
    SUBSET = "subset"


class BaseSelector(ABC):
    def __init__(self, result_type: ResultType, n_features: int):
        self._result_type = result_type
        self._n_features = n_features

        self._X = None
        self._support_mask = None
        self._selected = None
        self._rank = None
        self._weights = None
        self._fitted = False

    @abstractmethod
    def fit(X, y):
        raise NotImplementedError()

    def _check_fit(self):
        if not self._fitted:
            raise Exception("Model is not fitted!")

    def get_result_type(self):
        return self._result_type

    def get_features(self, k):
        self._check_fit()
        feats = self._X[:, self._support_mask]
        return feats[:, :k] if k and k < self._n_features else feats

    def get_selected(self):
        self._check_fit()
        if self._selected:
            return self._selected
        raise Exception("This selector does not return a subset of features!")

    def get_rank(self):
        self._check_fit()
        if self._rank:
            return self._rank
        raise Exception("This selector does not return feature ranks!")

    def get_weights(self):
        self._check_fit()
        if self._weights:
            return self._weights
        raise Exception("This selector does not return feature weights!")

    def get_top_k_rank(self, k):
        self._check_fit()
        if self._rank:
            if k < self._n_features:
                return self._rank[:k]
            else:
                raise ValueError("Given `k` should be lower than the number of selected `n_features!")
        raise Exception("This selector does not return feature ranks!")

    def get_top_k_weights(self, k):
        self._check_fit()
        if self._weights:
            if k < self._n_features:
                return self._weights[:k]
            else:
                raise ValueError("Given `k` should be lower than the number of selected `n_features`!")
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
