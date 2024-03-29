from .dataloader import DataLoader
from .shared_dataset import SharedDataset


class SharedDatasets:
    def __init__(self, dataloader: DataLoader):
        self._dataloader = dataloader
        self._datasets = {}

    def add_dataset(self, name, path):
        X, y, cols = self._dataloader.load(path, to_drop=['samples'])
        sd = SharedDataset(name, X, y, cols)
        self._datasets[name] = sd

    def add_datasets(self, paths_dict):
        for name, path in paths_dict.items():
            try:
                self.add_dataset(name, path)
                print(f"Added {name} dataset successfully")
            except Exception as e:
                print(f"Could not load {path}: {e}")

    def get_dataset(self, name):
        try:
            return self._datasets[name]
        except Exception:
            raise Exception(f"There is no dataset named `{name}`.")
