from sklearn.svm import SVC

from feature_selectors.base_models.recursive_feature_elimination import RFE


class SVMRFE(RFE):
    def __init__(self, n_features=None, verbose=0, **kwargs):
        super().__init__(SVC(kernel='linear', **kwargs), 'coef_', n_features=n_features, verbose=verbose)
