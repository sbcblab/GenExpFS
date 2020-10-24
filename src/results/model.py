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
