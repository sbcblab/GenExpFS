import csv
import os

from dataclasses import dataclass


@dataclass
class Result:
    name: str
    processing_time: float
    dataset_name: str
    num_features: int
    num_selected: int
    sampling: str
    result_type: str
    values: str

    def to_dict(self):
        return self.__dict__

    def fields(self):
        return self.__dict__.keys()


class ResultsWritter:
    def __init__(self, base_path):
        self._base_path = base_path

    def write(self, result):
        dir_path = os.path.join(self._base_path, result.name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        path_to_save = os.path.join(dir_path, f"{result.dataset_name}.csv",)
        with open(path_to_save, "a") as f:
            writer = csv.DictWriter(f, result.fields())

            if f.tell() == 0:
                writer.writeheader()

            writer.writerow(result.to_dict())
