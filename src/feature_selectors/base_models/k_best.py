from sklearn.feature_selection import SelectKBest


from .embedded import BaseEmbeddedFeatureSelector


class KBestFeatureSelector(BaseEmbeddedFeatureSelector):
    def __init__(self, score_func, n_features=None, p_threshold=0.05):
        model = SelectKBest(score_func, n_features, p_threshold)
        super().__init__(model, 'scores_')
