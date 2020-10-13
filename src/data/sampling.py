from time import time
import random

import numpy as np


def bootstrap(X, y=None):
    n_samples, _ = X.shape
    random.seed(time())
    np.random.seed(random.randint(0, int(time())))
    selected = np.random.choice(n_samples, n_samples)
    return (X[selected], y[selected]) if y is not None else X[selected]
