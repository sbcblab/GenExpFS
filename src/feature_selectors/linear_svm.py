from sklearn.svm import SVC


from feature_selectors.base_models.embedded import BaseEmbeddedFeatureSelector


class LinearSVMFeatureSelector(BaseEmbeddedFeatureSelector):
    def __init__(self, n_features=None, **kwargs):
        super().__init__(SVC(kernel='linear', **kwargs), 'coef_', n_features=n_features)
