from sklearn.svm import SVC

from .base_models.forward_feature_selector import ForwardFeatureSelector


class SVMForwardFeatureSelector(ForwardFeatureSelector):
    def __init__(self, n_features=None, cv_folds=5, verbose=0):
        super().__init__(SVC(), n_features, cv_folds, verbose)
