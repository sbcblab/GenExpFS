from numpy import np
from sklearn.preprocessing import label_binarize


class MultiClassModel():
    def __init__(self, model_cls, *args, **kwargs):
        self._model_cls = model_cls
        self._model_args = (args, kwargs)
        self._already_fit = False

    def _fit(self, X, y, importances, **kwargs):
        args_from_model, kwargs_from_model = self._model_args

        model = self._model_cls(*args_from_model, **kwargs_from_model)
        model.fit(X, y, **kwargs)

        coef = model.coef_ if hasattr(model, "coef_") else model.feature_importances_
        importances.append(coef)

    def fit(self, X, y, **kwargs):
        if self._already_fit:
            raise Exception('Model is already fit.')
        self._already_fit = True

        importances = []
        classes = np.unique(y)

        if len(classes) <= 2:
            self._fit(X, y, importances, **kwargs)
        else:
            binarized_classes = label_binarize(y, classes).transpose()
            for bin_y in binarized_classes:
                self._fit(X, bin_y, importances, **kwargs)

        self.feature_importances_ = np.array(importances).sum(axis=0)
        self.coef_ = self.feature_importances_
