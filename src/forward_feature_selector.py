import numpy as np
from sklearn.model_selection import KFold, cross_val_score


class ForwardFeatureSelector:
    def __init__(self, model, k, cv_folds=5):
        self._model = model
        self._k = k
        self._cv = KFold(cv_folds)
        self._selected = []

        self._fitted = False

    def _select_best(self, X, y, remaining):
        accuracies = np.array([
            cross_val_score(self._model, X[:, self._selected + [feat]], y, cv=self._cv, n_jobs=-1).mean()
            for feat
            in remaining
        ])

        best_idx = np.argmax(accuracies)
        best_feat = remaining[best_idx]

        return best_feat

    def fit(self, X, y, verbose=False):
        if self._fitted:
            raise Exception("Model is already fit.")
        self._fitted = True

        num_feats = X.shape[1]
        remaining = list(range(num_feats))
        self._support_mask = np.zeros(num_feats, dtype=bool)

        num_feats_to_select = np.min(np.array([num_feats, self._k]))
        for i in range(num_feats_to_select):
            selected_idx = self._select_best(X, y, remaining)

            self._selected.append(selected_idx)
            self._support_mask[selected_idx] = 1
            remaining.remove(selected_idx)

            if verbose:
                print(f'[{i+1}/{num_feats_to_select}] Selected variable {selected_idx}.')

        return self

    def get_suport(self, indices=False):
        return self._selected if indices else self._support_mask

    def transform(self, X):
        return X[:, self._selected]

    def fit_transform(self, X, y, verbose=False):
        return self.fit(X, y, verbose).transform(X)
