import numpy as np
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
