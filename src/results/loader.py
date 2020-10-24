import os

import pandas as pd

from feature_selectors.base_models import ResultType
from .sampling import SamplingType
from src.util.filesystem import files_in_dir_tree


class ResultsLoader:
    def __init__(self, base_path):
        if not os.path.exists(base_path):
            raise Exception(f"Folder '{base_path}' does not exist!")
        if not os.path.isdir(base_path):
            raise Exception(f"{base_path} is not a directory!")
        if os.listdir(base_path) == []:
            raise Exception(f"No results found at {base_path}!")
        self._base_path = base_path

    def get_results_names(self):
        return os.listdir(self._base_path)

    def load_by_name(self, name):
        alg_path = os.path.join(self._base_path, name)
        try:
            data_sets = os.listdir(alg_path)
        except FileNotFoundError:
            raise Exception(f"Directory {alg_path} not found!")
        if len(data_sets) == 0:
            raise Exception(f"No dataset found at {alg_path}")

        datasets = (pd.read_csv(os.path.join(alg_path, d)) for d in data_sets)
        return pd.concat(datasets, ignore_index=True)

    def load_by_dataset_name(self, dataset_name):
        if not dataset_name.endswith('.csv'):
            dataset_name += '.csv'

        paths = [p for p in files_in_dir_tree(self._base_path) if p.endswith(dataset_name)]
        if paths == []:
            raise Exception(f"No results found for dataset {dataset_name}")

        datasets = [pd.read_csv(p) for p in paths]
        return pd.concat(datasets, ignore_index=True)

    def load_all(self):
        paths = files_in_dir_tree(self._base_path)
        return pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)

    def load_by_sampling(self, sampling):
        sampling_types = {s.value for s in SamplingType}
        if sampling not in sampling_types:
            raise Exception(f"{sampling} must be one of: {sampling_types}")
        df = self.load_all()
        return df[df['sampling'] == sampling].reset_index(drop=True)

    def load_by_result_type(self, result_type):
        result_types = {rt.value for rt in ResultType}
        if result_type not in result_types:
            raise Exception(f"{result_type} must be one of: {result_types}")
        df = self.load_all()
        return df[df['result_type'] == result_type].reset_index(drop=True)
