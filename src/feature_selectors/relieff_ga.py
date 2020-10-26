from feature_selectors.base_models import FeatureSelectorPipeline
from .relieff import ReliefFFeatureSelector
from .svm_genetic_algorithm import SVMGAFeatureSelector


class ReliefFGAFeatureSelector(FeatureSelectorPipeline):
    def __init__(self, n_features, n_features_relieff=1000, **kwargs):
        selectors = [
            ReliefFFeatureSelector(n_features=n_features_relieff),
            SVMGAFeatureSelector(n_features=n_features)
        ]
        super().__init__(selectors)
