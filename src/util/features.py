import numpy as np
from sklearn.utils.multiclass import type_of_target


def _feature_type(X):
    t_dtype = str(X.dtype)
    is_numerical = 'float' in t_dtype or 'double' in t_dtype
    is_integer = 'int' in t_dtype or 'long' in t_dtype or 'short' in t_dtype

    if is_numerical:
        return 'continuous'

    t = type_of_target(X)

    def unique():
        return list(np.sort(np.unique(X)))

    if t == 'binary' and (is_integer and set(unique()) <= set([0, 1, -1]) or not is_integer):
        return 'categorical'

    elif t == 'multiclass':
        if is_integer:
            unique_values = unique()
            is_sequence = np.array([x + 1 == y for x, y in zip(unique_values, unique_values[1:])]).all()
            starts_at_0_or_1 = np.min(unique_values) in [0, 1]

            if is_sequence and starts_at_0_or_1:
                return 'categorical'
        else:
            return 'categorical'

    return 'continuous'


def feature_type(X):
    X = np.array(X)
    return _feature_type(X) if len(X.shape) == 1 else np.array([_feature_type(f) for f in X.T])


def is_discrete(X):
    return feature_type(X) == 'categorical'


def is_continuous(X):
    return feature_type(X) == 'continuous'
