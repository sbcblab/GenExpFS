from feature_selectors.base_models.k_best import KBestFeatureSelector
from evaluation.statistics import mutual_information


class MutualInformationFeatureSelector(KBestFeatureSelector):
    def __init__(self, n_features=None, p_threshold=0.05, **kwargs):
        super().__init__(mutual_information, n_features, p_threshold)
