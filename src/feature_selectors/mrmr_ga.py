from feature_selectors import SVMGAFeatureSelector, MRMRFeatureSelector
from feature_selectors.base_models import FeatureSelectorPipeline, Step


class MRMRGAFeatureSelector(FeatureSelectorPipeline):
    def __init__(self, n_features, n_features_mrmr=1000, **kwargs):
        steps = [
            Step(MRMRFeatureSelector, n_features=n_features_mrmr),
            Step(SVMGAFeatureSelector, n_features=n_features)
        ]
        super().__init__(steps)
