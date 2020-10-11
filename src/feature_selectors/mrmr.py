import numpy as np

from evaluation.statistics import mutual_information
from .base_selector import BaseSelector, ResultType


class MRMRFeatureSelector(BaseSelector):
    def __init__(self, n_features=None):
        super().__init__(ResultType.WEIGHTS, n_features)

    def _get_support_mask(self):
        if not self._fitted:
            raise Exception('Model is not fitted.')
        return self._support_mask

    def fit(self, X, y, verbose=0):
        if self._fitted:
            raise Exception('Model is already fitted.')

        self._X = X
        n_samples, n_features = X.shape

        if self._n_features > n_features:
            raise Exception(
                f'Number of features to select ({self._n_features}) is higher '
                f'than number of features in data ({n_features})'
            )

        selected = []
        remaining = list(range(n_features))

        selected_to_remaining_mi = np.zeros((self._n_features - 1, n_features))
        selected_to_class_mi = []

        features_to_class_mi = np.array(mutual_information(X, y))

        first_feature = np.argmax(features_to_class_mi)
        selected.append(first_feature)
        remaining.remove(first_feature)

        highest_mi = np.max(features_to_class_mi)
        selected_to_class_mi.append(highest_mi)

        if verbose:
            print(f'[1/{self._n_features}] Selected feat {first_feature} with mRMR: {highest_mi:0.3}')

        for num_selected in range(1, self._n_features):

            remaining_features = X[:, remaining]
            last_selected_feature = X.T[selected[-1]]
            last_selected_to_remaining_mi = mutual_information(remaining_features, last_selected_feature)

            selected_to_remaining_mi[num_selected - 1, remaining] = last_selected_to_remaining_mi

            redundancy = np.mean(selected_to_remaining_mi[:num_selected, remaining], axis=0)
            mRMR_scores = features_to_class_mi[remaining] - redundancy

            selected_idx = remaining[np.argmax(mRMR_scores)]
            selected.append(selected_idx)
            remaining.remove(selected_idx)

            higher_score = np.max(mRMR_scores)
            selected_to_class_mi.append(higher_score)

            if verbose:
                print(
                    f'[{num_selected+1}/{self._n_features}] Selected'
                    f' feat {selected_idx} with mRMR: {higher_score:0.3}'
                )

        self._support_mask = np.zeros(n_features, dtype=np.bool)
        self._support_mask[selected] = True
        self._selected = selected
        self._rank = selected
        self._wegihts = np.zeros(n_features, dtype='d')
        self._weights[selected] = selected_to_class_mi
        self._fitted = True

        return self
