from sklearn.tree import DecisionTreeClassifier


from feature_selectors.base_models.embedded import BaseEmbeddedFeatureSelector


class DecisionTreeFeatureSelector(BaseEmbeddedFeatureSelector):
    def __init__(self, n_features=None, **kwargs):
        super().__init__(DecisionTreeClassifier(**kwargs), 'feature_importances_', n_features=n_features)
