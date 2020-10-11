import os
from typing import List

import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, base_path):
        self._base_path = base_path

    def _load(
        self,
        path: str,
        targets: List[str],
        to_drop: List[str],
        check_columns: bool = True,
        ret_col_names: bool = True
    ):
        df = pd.read_csv(path)

        if check_columns:
            missing = []
            if len(targets) != 0:
                missing = [x for x in targets if x not in df.columns]
            missing += [x for x in to_drop if x not in df.columns]
            if len(missing) > 0:
                raise Exception(f"Field check: missing {missing} columns.")
        else:
            targets = [x for x in targets if x in df.columns]
            to_drop = [x for x in to_drop if x in df.columns]

        X = df.drop(targets + to_drop, axis=1).values

        to_return = (X,)

        if len(targets) > 0:
            targets = targets[0] if len(targets) == 1 else targets
            y = df[targets].values
            to_return += (y,)

        if ret_col_names:
            col_names = np.array([x for x in df.columns if x not in targets and x not in to_drop])
            to_return += (col_names,)

        return to_return

    def recursive_load(
        self,
        path: str,
        targets: List[str] = ['type', 'class'],
        to_drop: List[str] = [],
        check_columns: bool = False,
        ret_col_names: bool = True
    ):
        full_path = os.path.join(self._base_path, path)

        for f in os.listdir(full_path):
            file_path = os.path.join(full_path, f)

            if os.path.isdir(file_path):
                yield from self.recursive_load(file_path, targets, to_drop)
            else:
                yield self._load(file_path, targets, to_drop)

    def load(
        self,
        path: str,
        targets: List[str] = ['type', 'class'],
        to_drop: List[str] = [],
        check_columns: bool = False,
        ret_col_names: bool = True
    ):
        full_path = os.path.join(self._base_path, path)

        if os.path.isdir(full_path):
            raise Exception("Given dataset path is a dir, use `recursive_load` function instead.")
        elif not full_path.endswith('.csv'):
            raise Exception("Only .csv format currently supported.")
        else:
            return self._load(full_path, targets, to_drop, check_columns, ret_col_names)

    def load_paths(
        self,
        paths: List[str],
        targets: List[str] = ['type', 'class'],
        to_drop: List[str] = [],
        check_columns: bool = False,
        ret_col_names: bool = True
    ):
        for path in paths:
            yield self.load(path, targets, to_drop, check_columns, ret_col_names)
