import numpy as np
from sklearn.preprocessing import label_binarize

from .base_selector import BaseSelector, ResultType


class MultiClassModel(BaseSelector):

    result_type = ResultType.WEIGHTS

    def __init__(self, model_cls: BaseSelector, n_features=None, *args, **kwargs):
        super().__init__(n_features)
        self._model_cls = model_cls
        self._model_args = (args, kwargs)

    def _fit(self, X, y, all_weights, **kwargs):
        args_from_model, kwargs_from_model = self._model_args

        model = self._model_cls(*args_from_model, **kwargs_from_model)
        model.fit(X, y, **kwargs)

        features_weights = model.get_weights()
        all_weights = all_weights.append(features_weights)

    def fit(self, X, y, **kwargs):
        self.check_already_fitted()
        self._fitted = True

        all_weights = []
        classes = np.unique(y)

        if len(classes) <= 2:
            self._fit(X, y, all_weights, **kwargs)
        else:
            binarized_classes = label_binarize(y, classes).transpose()
            for bin_y in binarized_classes:
                self._fit(X, bin_y, all_weights, **kwargs)

        self._weights = np.array(all_weights).mean(axis=0)
        self._rank = np.argsort(self._weights)[::-1]
        return self
