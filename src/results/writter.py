import csv
import os

import pandas as pd

from .model import Result


class ResultsWritter:
    def __init__(self, base_dir, file_name='selection.csv'):
        self._base_dir = base_dir
        self._file_name = file_name

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        elif not os.path.isdir(base_dir):
            raise Exception(f"base_dir[{base_dir}] must be a directory!")

    def _write_csv(self, path_to_save, result: Result):
        with open(path_to_save, "a") as f:
            writer = csv.DictWriter(f, result.fields())

            if f.tell() == 0:
                writer.writeheader()

            writer.writerow(result.to_dict())

    def write_csv(self, result: Result):
        if not os.path.exists(self._base_dir):
            os.makedirs(self._base_dir)

        file_name = self._file_name if self._file_name.endswith('.csv') else f"{self._file_name}.csv"
        path_to_save = os.path.join(self._base_dir, file_name)

        self._write_csv(path_to_save, result)

    def write_dataframe(self, df: pd.DataFrame, file_name: str):
        if not os.path.exists(self._base_dir):
            os.makedirs(self._base_dir)

        path = os.path.join(self._base_dir, file_name)

        if os.path.exists(path):
            df.to_csv(path, mode='a', index=False, header=False)
        else:
            df.to_csv(path, index=False)
