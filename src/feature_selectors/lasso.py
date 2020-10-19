from sklearn.linear_model import Lasso


from feature_selectors.base_models.embedded import BaseEmbeddedFeatureSelector


class LassoFeatureSelector(BaseEmbeddedFeatureSelector):
    def __init__(self, n_features=None, **kwargs):
        super().__init__(Lasso(alpha=0.001, **kwargs), 'coef_', n_features=n_features)
