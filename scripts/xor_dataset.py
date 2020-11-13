from itertools import cycle, chain
import os

import numpy as np


DATASETS_PATH = 'datasets/xor'
np.random.seed(1)


class DatasetParams:
    def __init__(self, n_samples: int, n_features: int):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_noisy = n_features - 2

    def name(self):
        return f"xor_{self.n_samples}samples_{self.n_features}features"

    def csv_name(self):
        return self.name() + '.csv'

    def feature_names(self):
        types = chain(
            ("informative" for _ in range(2)),
            ("noisy" for _ in range(self.n_features - 2))
        )
        return [f'{t}_{x}' for t, x in zip(types, range(self.n_features))]

    def build_dataset(self):
        instance_model = cycle([([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)])
        X, y = zip(*[(x, y) for i, (x, y) in zip(range(self.n_samples), instance_model)])
        noisy_features = np.random.randint(2, size=(self.n_samples, self.n_noisy))
        X = np.concatenate((X, noisy_features), axis=1)
        return X, y, self.feature_names()

    def to_csv(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        X, y, cols = self.build_dataset()
        data = np.column_stack((X, y))
        sepcs = ['%i' for _ in cols] + ['%i']
        cols += ['class']
        header = ','.join(f'"{c}"' for c in cols)
        path_to_save = os.path.join(path, self.csv_name())
        np.savetxt(
            path_to_save,
            data,
            fmt=sepcs,
            header=header,
            delimiter=',',
            comments=''
        )
        print(f"Saved dataset to {path_to_save}")


dataset = DatasetParams(n_samples=500, n_features=50)
dataset.to_csv(DATASETS_PATH)
