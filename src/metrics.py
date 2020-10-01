import numpy as np
from scipy.spatial.distance import jaccard
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


def one_vs_all_roc_auc(y, y_pred):
    classes = np.unique(y)
    bin_true = label_binarize(y, classes).transpose()
    bin_pred = label_binarize(y_pred, classes).transpose()

    scores = [
        roc_auc_score(tr, pr)
        for tr, pr
        in zip(bin_true, bin_pred)
    ]

    return np.mean(scores)


def jaccard_score(s1, s2):
    is_list = isinstance(s1, list) and isinstance(s2, list) or isinstance(s1, np.ndarray) and isinstance(s1, np.ndarray)

    if is_list and len(s1) == len(s2):
        return 1 - jaccard(s1, s2)
    elif isinstance(s1, set) and isinstance(s2, set):
        intersection = s1.intersection(s2)
        union = s1.union(s2)
        return len(intersection) / len(union)
    else:
        raise TypeError("Only a pair of `sets`, `lists` or `numpy arrays` is allowed.")
