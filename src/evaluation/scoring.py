from sklearn.metrics import make_scorer, accuracy_score, recall_score, precision_score, f1_score

from metrics import one_vs_all_roc_auc


default_scoring = {
    "accuracy": make_scorer(accuracy_score),
    "macro_recall": make_scorer(recall_score, average='macro'),
    "macro_precision": make_scorer(precision_score, average='macro', zero_division=False),
    "micro_f1": make_scorer(f1_score, average='micro'),
    "macro_f1": make_scorer(f1_score, average='macro'),
    "roc_auc": make_scorer(one_vs_all_roc_auc)
}
