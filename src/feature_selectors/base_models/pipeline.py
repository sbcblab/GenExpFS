import numpy as np


from .base_selector import BaseSelector, ResultType


class Step:
    def __init__(self, feature_selector: BaseSelector, *args, **kwargs):
        self.feature_selector = feature_selector
        self.args = args
        self.kwargs = kwargs


def check_steps(steps):
    all_steps_have_n_features = all([s.feature_selector._n_features for s in steps])

    if not all_steps_have_n_features:
        raise Exception("All steps must have n_features!")

    num_features_decreases = all([
        s1.feature_selector._n_features > s2.feature_selector._n_features
        for s1, s2 in zip(steps, steps[1:])
    ])

    if not num_features_decreases:
        raise Exception("n_features should decrease along the pipeline!")


class FeatureSelectorPipeline(BaseSelector):
    def __init__(self, steps: list):
        last_step = steps[-1]
        result_type = last_step.feature_selector._result_type
        n_features = last_step._n_features
        super().__init__(result_type, n_features)

        check_steps(steps)
        self._steps = steps

    def fit(self, X, y, **kwargs):
        self.check_already_fitted()
        self._X = X
        n_samples, n_features = X.shape

        X_selected = X
        selected_idx = list(range(n_features))

        for step in self._steps:
            fs = step.feature_selector(*step.args, **step.kwargs)
            fs.fit(X_selected, y)

            subset_indices = fs.get_features()
            selected_idx = selected_idx[subset_indices]

            X_selected = fs.get_features()

        self._selected = selected_idx

        self._support_mask = np.zeros(n_features)
        self._support_mask[selected_idx] = True

        if fs._result_type is ResultType.WEIGHTS:
            self._weights = np.zeros(n_features)
            self._weights[selected_idx] = fs.get_weights()
            self._rank = np.argsort(self._weights)[::-1]

        elif fs._result_type is ResultType.RANK:
            self._rank = fs.get_rank()

        self._fitted = True
        return self
