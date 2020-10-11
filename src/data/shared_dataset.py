from multiprocessing.sharedctypes import RawArray
from ctypes import c_char

from numpy import np

from .dataset import Dataset


class SharedDataset(Dataset):
    def __init__(self, path, data, classes, columns):
        self._data_shape = data.shape
        self._data_type = data.dtype
        self._classes_shape = classes.shape
        self._classes_type = classes.type
        self._columns_shape = columns.shape
        self._columns_type = columns.type

        c_data = RawArray(np.ctypeslib.as_ctypes_type(data.dtype), data.size)
        c_classes = RawArray(c_char, classes.nbytes)
        c_columns = RawArray(c_char, columns.nbytes)

        np_data = np.frombuffer(c_data, dtype=data.dtype).reshape(data.shape)
        np_classes = np.frombuffer(c_classes, dtype=classes.dtype).reshape(classes.shape)
        np_columns = np.frombuffer(c_columns, dtype=columns.dtype).reshape(columns.shape)

        np.copyto(np_data, data)
        np.copyto(np_classes, classes)
        np.copyto(np_columns, columns)

        super().__init__(path, c_data, c_classes, c_columns)

    def get(self):
        data = np.frombuffer(self.data, dtype=self._data_type).reshape(self._data_shape)
        classes = np.frombuffer(self.classes, dtype=self._classes_type).reshape(self._classes_shape)
        columns = np.frombuffer(self.columns, dtype=self._columns_type).reshape(self._columns_shape)
        return data, classes, columns

    def get_instances(self):
        return np.frombuffer(self.data, dtype=self._data_type).reshape(self._data_shape)

    def get_classes(self):
        return np.frombuffer(self.classes, dtype=self._classes_type).reshape(self._classes_shape)

    def get_column_names(self):
        return np.frombuffer(self.columns, dtype=self._columns_type).reshape(self._columns_shape)
