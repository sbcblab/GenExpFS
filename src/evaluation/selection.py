from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import minmax_scale

from scoring import default_scoring
from models import default_models


class SelectionEvaluator:
    def __init__(self, models=default_models, scoring=default_scoring):
        self._models = models
        self._scoring = scoring

    def _eval(self, X, y, model):
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
