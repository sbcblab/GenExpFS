from .metrics import one_vs_all_roc_auc

from sklearn.linear_model import SVC
from sklearn.metrics import make_scorer, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import minmax_scale


def classification_eval(X, y):
    X = minmax_scale(X)
    t_svc = SVC()

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "macro_recall": make_scorer(recall_score, average='macro'),
        "macro_precision": make_scorer(precision_score, average='macro', zero_division=False),
        "micro_f1": make_scorer(f1_score, average='micro'),
        "macro_f1": make_scorer(f1_score, average='macro'),
        "roc_auc": make_scorer(one_vs_all_roc_auc)
    }

    cv = StratifiedKFold()
    results = cross_validate(t_svc, X, y, cv=cv, scoring=scoring)
    avg_results = {k.replace("test_", ""): v.mean() for (k, v) in results.items() if not k.endswith("time")}

    return avg_results
