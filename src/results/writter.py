import csv
import os

from .model import Result


class ResultsWritter:
    def __init__(self, base_path, file_name='selection'):
        self._base_path = base_path
        self._file_name = file_name

    def _write_csv(path_to_save, result):
        with open(path_to_save, "a") as f:
            writer = csv.DictWriter(f, result.fields())

            if f.tell() == 0:
                writer.writeheader()

            writer.writerow(result.to_dict())

    def write_csv_per_model_and_dataset(self, result: Result):
        dir_path = os.path.join(self._base_path, result.name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        path_to_save = os.path.join(dir_path, f"{result.dataset_name}.csv",)

        self._write_csv(path_to_save, result)

    def write_csv(self, result: Result):
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

        path_to_save = os.path.join(self._base_path, f"{self._file_name}.csv",)

        self._write_csv(path_to_save, result)
