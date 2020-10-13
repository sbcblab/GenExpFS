from sklearn.feature_selection import SelectKBest as SKSelectKBest


from .embedded import BaseEmbeddedFeatureSelector


class SelectKBest(BaseEmbeddedFeatureSelector):
    def __init__(self, score_func, n_features=None, p_threshold=0.05):
        model = SKSelectKBest(score_func, n_features, p_threshold)
        super().__init__(model, 'scores_')
