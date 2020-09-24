from numpy import np
from sklearn.preprocessing import label_binarize


class MultiClassModel():
    def __init__(self, model_cls, *args, **kwargs):
        self._model_cls = model_cls
        self._model_args = (args, kwargs)
        self._models = []
        self._fit = False

    def fit(self, X, y, sample_weight=None, check_input=True):
        if self._fit:
            raise Exception('Model is already fit.')
        self._fit = True

        importances = []
        classes = np.unique(y)
        binarized_classes = label_binarize(y, classes).transpose()

        for bin_y in binarized_classes:
            args, kwargs = self._model_args
            model = self._model_cls(*args, **kwargs)
            model.fit(X, bin_y)
            _importances = model.coef_ if hasattr(model, "coef_") else model.feature_importances_
            importances.append(_importances)

        self.feature_importances_ = np.array(importances)
        self.coef_ = self.feature_importances_
