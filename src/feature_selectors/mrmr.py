import numpy as np

from statistics import mutual_information


class MRMRFeatureSelector():
    def __init__(self, k):
        self._k = k

        self._support_mask = None
        self._fitted = False

    def _get_support_mask(self):
        if not self._fitted:
            raise Exception('Model is not fitted.')
        return self._support_mask

    def fit(self, X, y, verbose=0):
        if self._fitted:
            raise Exception('Model is already fitted.')

        n_samples, n_features = X.shape

        if self._k > n_features:
            raise Exception(
                f'Number of features to select ({self._k}) is higher than number of features in data ({n_features})'
            )

        selected = []
        remaining = list(range(n_features))

        selected_to_remaining_mi = np.zeros((self._k - 1, n_features))
        selected_to_class_mi = []

        features_to_class_mi = np.array(mutual_information(X, y))

        first_feature = np.argmax(features_to_class_mi)
        selected.append(first_feature)
        remaining.remove(first_feature)

        highest_mi = np.max(features_to_class_mi)
        selected_to_class_mi.append(highest_mi)

        if verbose:
            print(f'[1/{self._k}] Selected feat {first_feature} with mRMR: {highest_mi:0.3}')

        for num_selected in range(1, self._k):

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
                print(f'[{num_selected+1}/{self._k}] Selected feat {selected_idx} with mRMR: {higher_score:0.3}')

        self._support_mask = np.zeros(n_features, dtype=np.bool)
        self._support_mask[selected] = True
        self.ranking_ = selected
        self.mrmr_ = selected_to_class_mi
        self._fitted = True

        return self

    def get_suport(self, indices=False):
        return self.ranking_ if indices else self._support_mask

    def transform(self, X):
        return X[:, self.ranking_]

    def fit_transform(self, X, y, verbose=0):
        return self.fit(X, y, verbose).transform(X)
