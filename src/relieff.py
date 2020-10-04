import numpy as np
from sklearn.neighbors import NearestNeighbors


class ReliefF:
    def __init__(self, k=20, n_neighbors=10):
        self._k = k
        self._n = n_neighbors

    def _n_first_x_in_y(self, x, y):
        num_x = len(x)
        i, first_k = 0, []
        while len(first_k) < self._n and i < num_x:
            if x[i] in y:
                first_k.append(x[i])
            i += 1

        return first_k

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.feature_importances_ = np.zeros((n_features))

        _, neighbors = NearestNeighbors(n_samples, metric='manhattan').fit(X).kneighbors(X)

        classes, classes_counts = np.unique(y, return_counts=True)
        class_prob = {classes[n]: x for n, x in enumerate(classes_counts / n_samples)}

        class_factor = {k: {c: class_prob[k] / (1 - class_prob[c]) for c in classes if c != k} for k in classes}

        for count in classes_counts:
            if count < self._n:
                raise Exception("Number of instances of a class can't be lower than k")

        instances_by_class = {x: np.arange(n_samples)[y == x] for x in classes}

        for instance in range(n_samples):
            instance_class = y[instance]

            instance_neighbors = neighbors[instance][1:]

            hits = self._n_first_x_in_y(instance_neighbors, instances_by_class[instance_class])

            misses_by_class = {
                c: self._n_first_x_in_y(instance_neighbors, instances_by_class[c])
                for c in classes
                if c != instance_class
            }

            for hit in hits:
                self.feature_importances_ -= np.array(abs(X[instance] - X[hit])) / self._n

            for (c, misses) in misses_by_class.items():
                for miss in misses:
                    class_weight = class_factor[instance_class][c]
                    miss_weights = np.array(abs(X[instance] - X[miss])) / (self._n * class_weight)
                    self.feature_importances_ += miss_weights

        return self
