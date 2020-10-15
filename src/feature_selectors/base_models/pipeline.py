import numpy as np


from .base_selector import BaseSelector, ResultType


def check_selectors(selectors):
    all_selectors_have_n_features = all([s._n_features for s in selectors])

    if not all_selectors_have_n_features:
        raise Exception("All selectors must have n_features param not None!")

    num_features_decreases = all([
        s1._n_features > s2._n_features
        for s1, s2 in zip(selectors, selectors[1:])
    ])

    if not num_features_decreases:
        raise Exception("n_features should decrease along the pipeline!")


class FeatureSelectorPipeline(BaseSelector):

    result_type = ResultType.SUBSET

    def __init__(self, selectors: list):
        last_selector = selectors[-1]
        super().__init__(last_selector._n_features)

        check_selectors(selectors)
        self.result_type = last_selector.result_type
        self._selectors = selectors

    def fit(self, X, y, **kwargs):
        self.check_already_fitted()
        self._X = X
        n_samples, n_features = X.shape

        X_selected = X
        selected_idx = np.arange(n_features)

        for fs in self._selectors:
            fs.fit(X_selected, y)

            subset_indices = np.array(fs.get_selected())
            selected_idx = selected_idx[subset_indices]

            X_selected = fs.get_features()

        self._selected = selected_idx

        self._support_mask = np.zeros(n_features)
        self._support_mask[selected_idx] = True

        if self.result_type is ResultType.WEIGHTS:
            self._weights = np.zeros(n_features, dtype=np.bool)
            self._weights[selected_idx] = fs.get_weights()
            self._rank = np.argsort(self._weights)[::-1]

        elif self.result_type is ResultType.RANK:
            self._rank = fs.get_rank()

        self._fitted = True
        return self
