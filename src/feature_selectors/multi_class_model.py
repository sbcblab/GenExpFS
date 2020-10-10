import numpy as np
from sklearn.preprocessing import label_binarize

from .base_selector import BaseSelector, ResultType


class MultiClassModel(BaseSelector):
    def __init__(self, model_cls: BaseSelector, *args, **kwargs):
        super().__init__(ResultType.COMPLETE_WEIGHTS)
        self._model_cls = model_cls
        self._model_args = (args, kwargs)

    def _fit(self, X, y, all_weights, **kwargs):
        args_from_model, kwargs_from_model = self._model_args

        model = self._model_cls(*args_from_model, **kwargs_from_model)
        model.fit(X, y, **kwargs)

        features_weights = model.get_weights()
        all_weights = all_weights.append(features_weights)

    def fit(self, X, y, **kwargs):
        if self._fitted:
            raise Exception('Model is already fit.')
        self._fitted = True

        all_weights = []
        classes = np.unique(y)

        if len(classes) <= 2:
            self._fit(X, y, all_weights, **kwargs)
        else:
            binarized_classes = label_binarize(y, classes).transpose()
            for bin_y in binarized_classes:
                self._fit(X, bin_y, all_weights, **kwargs)

        self._weights = np.array(all_weights).sum(axis=0)
        self._rank = np.argsort(self._weights)[::-1]
        return self
