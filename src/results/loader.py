import os

import pandas as pd

from feature_selectors.base_models import ResultType
from data.sampling import SamplingType
from util.filesystem import files_in_dir_tree


class ResultsLoader:
    def __init__(self, results_path):
        if not os.path.exists(results_path):
            raise Exception(f"'{results_path}' does not exist!")

        if os.path.isdir(results_path):
            self._is_dir = True
            if [f for f in files_in_dir_tree(results_path) if f.endswith('.csv')] == []:
                raise Exception(f"No results found at {results_path}!")

        elif not results_path.endswith('.csv'):
            raise Exception(f"{results_path} should be either a path to dir "
                            f"containing csv files or a path to csv itself.")

        self._results_path = results_path

    def load_all(self):
        if self._is_dir:
            paths = files_in_dir_tree(self._results_path)
            return pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
        else:
            return pd.read_csv(self._results_path)

    def load_by(self, field, field_value, allowed_field_values=None):
        if allowed_field_values is not None and field_value not in allowed_field_values:
            raise Exception(f"{field_value} must be one of: {allowed_field_values}")

        df = self.load_all()
        results = df[df[field] == field_value].reset_index(drop=True)

        if results.size == 0:
            raise Exception(f"No results found for {field} {field_value}")

        return results

    def load_by_sampling(self, sampling):
        sampling_types = {s.value for s in SamplingType}
        return self.load_by('sampling', sampling, allowed_field_values=sampling_types)

    def load_by_result_type(self, result_type):
        result_types = {rt.value for rt in ResultType}
        return self.load_by('result_type', result_type, allowed_field_values=result_types)

    def load_by_dataset(self, dataset_name):
        return self.load_by('dataset_name', dataset_name)

    def load_by_name(self, name):
        return self.load_by('name', name)
