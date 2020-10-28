from feature_selectors.base_models.k_best import KBestFeatureSelector
from evaluation.statistics import kruskal_wallis


class KruskalWallisFeatureSelector(KBestFeatureSelector):
    def __init__(self, n_features=None, **kwargs):
        super().__init__(kruskal_wallis, n_features)
