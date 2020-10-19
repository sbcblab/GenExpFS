from sklearn.ensemble import RandomForestClassifier


from feature_selectors.base_models.embedded import BaseEmbeddedFeatureSelector


class RandomForestFeatureSelector(BaseEmbeddedFeatureSelector):
    def __init__(self, n_features=None, **kwargs):
        super().__init__(RandomForestClassifier(**kwargs), 'feature_importances_', n_features=n_features)
