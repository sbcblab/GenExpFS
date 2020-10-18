from sklearn.linear_model import LogisticRegression

from .base_models.forward_feature_selector import ForwardFeatureSelector


class LRForwardFeatureSelector(ForwardFeatureSelector):
    def __init__(self, n_features=None, cv_folds=5, verbose=0):
        super().__init__(LogisticRegression(max_iter=300), n_features, cv_folds, verbose)
