import numpy as np
from scipy.spatial.distance import jaccard, hamming, dice
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


def normalized_hamming_distance(a, b):
    try:
        return hamming(a, b)
    except Exception:
        raise TypeError("Only pair of `lists` or `numpy arrays` of the same size is allowed.")


def set_normalized_hamming_distance(a, b):
    if isinstance(a, set) and isinstance(b, set):
        return len(a.symmetric_difference(b)) / len(a.union(b))
    else:
        raise TypeError("A and B must be pair of `sets`")


def dice_coefficient(a, b):
    is_list = isinstance(a, list) and isinstance(b, list) or isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
    if is_list and len(a) == len(b):
        return 1 - dice(a, b)
    elif isinstance(a, set) and isinstance(b, set):
        return 2 * len(a.intersection(b)) / (len(a) + len(b))
    else:
        raise TypeError("Only a pair of `sets`, `lists` or `numpy arrays` is allowed.")


def ochiai_index(a, b):
    if isinstance(a, set) and isinstance(b, set):
        intersection = a.intersection(b)
        return len(intersection) / np.sqrt(len(a) * len(b))
    else:
        raise TypeError("Only a pair of `sets` is allowed.")


def kuncheva_index(a, b, m):
    '''
    m = total number of features
    k = length of features - assumes equal length of a and b
    r = number of common elements in both signatures
    kuncheva_index = (r*m - k^2)/k*(m-k)
    '''

    feat_size = len(a)
    same_size = feat_size == len(b)
    if m == feat_size:
        return 0

    if same_size and isinstance(a, set) and isinstance(b, set):
        r = a.intersection(b)
        k = feat_size
        return (len(r) * m - np.power(k, 2)) / (k * (m - k))
    else:
        raise TypeError("Only a pair of `sets` of the same size is allowed.")


def percentage_of_overlapping_features(a, b):
    if isinstance(a, set) and isinstance(b, set):
        intersection = a.intersection(b)
        return len(intersection) / len(a)
    else:
        raise TypeError("Only a pair of `sets` is allowed.")
