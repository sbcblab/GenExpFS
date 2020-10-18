from itertools import chain
import os

from dataclasses import dataclass, field
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import minmax_scale


DATASETS_PATH = 'datasets/synthetic'
random_state = np.random.RandomState(1)


@dataclass
class DatasetParams:
    n_samples: int
    n_features: int
    n_informative: int = 0
    n_redundant: int = 0
    n_repeated: int = 0
    n_noisy: int = field(init=False)

    def __post_init__(self):
        self.n_noisy = self.n_features - self.n_informative - self.n_redundant - self.n_repeated

    def name(self):
        _name = f"synth_{self.n_samples}samples_{self.n_features}features"
        if self.n_informative > 0:
            _name += f"_{self.n_informative}informative"
        if self.n_redundant > 0:
            _name += f"_{self.n_redundant}redundant"
        if self.n_repeated > 0:
            _name += f"_{self.n_repeated}repeated"
        return _name

    def csv_name(self):
        return self.name() + '.csv'

    def feature_names(self):
        types = chain(
            ("informative" for _ in range(self.n_informative)),
            ("redundant" for _ in range(self.n_redundant)),
            ("repeated" for _ in range(self.n_repeated)),
            ("noisy" for _ in range(self.n_noisy))
        )
        return [f'{t}_{x}' for t, x in zip(types, range(self.n_features))]

    def build_dataset(self):
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=self.n_informative,
            n_redundant=self.n_redundant,
            n_repeated=self.n_repeated,
            shuffle=False,
            random_state=random_state
        )
        return minmax_scale(X), y, self.feature_names()

    def to_csv(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        X, y, cols = self.build_dataset()
        data = np.column_stack((X, y))
        sepcs = ['%.5f' for _ in cols] + ['%i']
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


datasets = [
    DatasetParams(n_samples=100, n_features=5000, n_informative=50),
    DatasetParams(n_samples=100, n_features=5000, n_informative=50, n_redundant=50),
    DatasetParams(n_samples=100, n_features=5000, n_informative=50, n_repeated=50),
    DatasetParams(n_samples=100, n_features=5000, n_informative=50, n_repeated=50, n_redundant=50),
    DatasetParams(n_samples=200, n_features=5000, n_informative=50),
    DatasetParams(n_samples=200, n_features=5000, n_informative=50, n_redundant=50),
    DatasetParams(n_samples=200, n_features=5000, n_informative=50, n_repeated=50),
    DatasetParams(n_samples=200, n_features=5000, n_informative=50, n_repeated=50, n_redundant=50),
]

[dataset.to_csv(DATASETS_PATH) for dataset in datasets]
