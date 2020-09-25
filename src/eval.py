from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import minmax_scale


class FeaturesEvaluator:
    def __init__(self, models={}, scoring={}, stability={}):
        self._models = models
        self._scoring = scoring
        self._stability = stability

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

    def classification_eval(self, X, y):
        return {name: self._eval(X, y, model) for name, model in self._models.items()}
