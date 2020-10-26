import csv
import os

import pandas as pd

from .model import Result


class ResultsWritter:
    def __init__(self, base_dir):
        self._base_dir = base_dir

        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        elif not os.path.isdir(base_dir):
            raise Exception(f"base_dir[{base_dir}] must be a directory!")

    def write_dict(self, d, file_name):
        path_to_save = os.path.join(self._base_dir, file_name)
        file_name = file_name if file_name.endswith('.csv') else f"{file_name}.csv"
        with open(path_to_save, "a") as f:
            writer = csv.DictWriter(f, d.keys())

            if f.tell() == 0:
                writer.writeheader()

            writer.writerow(d)

    def write_result(self, result: Result, file_name: str):
        self.write_dict(result.to_dict(), file_name)

    def write_dataframe(self, df: pd.DataFrame, file_name: str):
        file_name = file_name if file_name.endswith('.csv') else f"{file_name}.csv"
        path_to_save = os.path.join(self._base_dir, file_name)

        if os.path.exists(path_to_save):
            df.to_csv(path_to_save, mode='a', index=False, header=False)
        else:
            df.to_csv(path_to_save, index=False)
