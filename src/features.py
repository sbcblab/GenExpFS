import numpy as np
from sklearn.utils.multiclass import type_of_target


def _feature_type(X):
    t = type_of_target(X)
    unique = list(np.sort(np.unique(X)))
    t_dtype = str(X.dtype)

    if t == 'binary' and 'int' in t_dtype and set(unique) <= set([0, 1, -1]):
        return 'categorical'

    elif t == 'multiclass':
        if 'int' in t_dtype:
            is_sequence = np.array([x + 1 == y for x, y in zip(unique, unique[1:])]).all()
            starts_at_0_or_1 = np.min(unique) in [0, 1]

            if is_sequence and starts_at_0_or_1:
                return 'categorical'
        elif 'float' not in t_dtype:
            return 'categorical'

    return 'continuous'


def feature_type(X):
    X = np.array(X)
    return _feature_type(X) if len(X.shape) == 1 else np.array([_feature_type(f) for f in X.T])


def is_discrete(X):
    return feature_type(X) == 'categorical'


def is_continuous(X):
    return feature_type(X) == 'continuous'
