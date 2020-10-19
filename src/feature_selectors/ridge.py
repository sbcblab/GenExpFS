from sklearn.linear_model import RidgeClassifier


from feature_selectors.base_models.embedded import BaseEmbeddedFeatureSelector


class RidgeClassifierFeatureSelector(BaseEmbeddedFeatureSelector):
    def __init__(self, n_features=None, **kwargs):
        super().__init__(RidgeClassifier(**kwargs), 'coef_', n_features=n_features)
