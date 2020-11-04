from enum import Enum
from time import time
import random

import numpy as np


class SamplingType(Enum):
    NONE = 'none'
    BOOTSTRAP = 'bootstrap'
    PERCENT_90 = 'percent90'


def bootstrap(X, y=None):
    n_samples, _ = X.shape
    random.seed(time())
    np.random.seed(random.randint(0, int(time())))
    selected = np.random.choice(n_samples, n_samples)
    return (X[selected], y[selected]) if y is not None else X[selected]


def percent90(X, y=None):
    n_samples, _ = X.shape
    random.seed(time())
    np.random.seed(random.randint(0, int(time())))
    selected = np.random.choice(n_samples, int(n_samples * 0.9), replace=False)
    return (X[selected], y[selected]) if y is not None else X[selected]
