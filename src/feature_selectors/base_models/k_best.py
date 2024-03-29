from sklearn.feature_selection import SelectKBest


from .embedded import BaseEmbeddedFeatureSelector


class KBestFeatureSelector(BaseEmbeddedFeatureSelector):
    def __init__(self, score_func, n_features=None):
        sklearn_n_features = n_features if n_features else 'all'
        model = SelectKBest(score_func, sklearn_n_features)
        super().__init__(model, 'scores_', n_features=n_features)
