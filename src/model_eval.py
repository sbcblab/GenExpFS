from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import minmax_scale

from .default import evaluation_scoring, evaluation_models


class ModelEvaluator:
    def __init__(self, models=evaluation_models, scoring=evaluation_scoring):
        self._models = models
        self._scoring = scoring

    def _eval(self, X, y, model, scoring):
        X = minmax_scale(X)
        cv = StratifiedKFold()

        results = cross_validate(model, X, y, cv=cv, scoring=self._scoring)

        avg_results = {
            k.replace("test_", ""): v.mean()
            for (k, v) in results.items()
            if not k.endswith("time")
        }

        return avg_results

    def eval(self, X, y):
        return {name: self._eval(X, y, model) for name, model in self._models.items()}
