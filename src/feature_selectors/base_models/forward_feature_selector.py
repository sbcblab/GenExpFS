import numpy as np
from sklearn.model_selection import KFold, cross_val_score

from .base_selector import BaseSelector, ResultType


class ForwardFeatureSelector(BaseSelector):
    def __init__(self, model, n_features=None, cv_folds=5, verbose=0):
        super().__init__(ResultType.RANK, n_features)
        self._model = model
        self._cv = KFold(cv_folds)
        self._verbose = verbose
        self._selected = []

    def _select_best(self, X, y, remaining):
        accuracies = np.array([
            cross_val_score(self._model, X[:, self._selected + [feat]], y, cv=self._cv).mean()
            for feat
            in remaining
        ])

        best_idx = np.argmax(accuracies)
        best_feat = remaining[best_idx]

        return best_feat

    def fit(self, X, y):
        self.check_already_fitted()
        self._fitted = True
        self._X = X

        num_feats = X.shape[1]
        remaining = list(range(num_feats))
        self._support_mask = np.zeros(num_feats, dtype=bool)

        n_features = num_feats if self._n_features is None else self._n_features
        num_feats_to_select = np.min(np.array([num_feats, n_features]))

        for i in range(num_feats_to_select):
            selected_idx = self._select_best(X, y, remaining)

            self._selected.append(selected_idx)
            self._support_mask[selected_idx] = 1
            remaining.remove(selected_idx)

            if self._verbose:
                print(f'[{i+1}/{num_feats_to_select}] Selected variable {selected_idx}.')

        self._rank = np.copy(self._selected)
        return self
