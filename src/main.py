from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import Lasso, RidgeClassifier
from sklearn.metrics import make_scorer, accuracy_score, recall_score, precision_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .metrics import one_vs_all_roc_auc
from .multi_class_model import MultiClassModel
from .select_k_significant_best import SelectKSignificantBest
from .statistics import kruskal_wallis

filtering_strategies = [
    (SelectKBest, 'SelectKBeast'),
    (SelectKSignificantBest, 'SelectKSignificantBest')
]

filtering_functions = [
    (f_classif, 'Anova'),
    (mutual_info_classif, 'MultualInformation'),
    (kruskal_wallis, 'KruskalWallis')
]

embedded_selection = {
    'DecisionTree': DecisionTreeClassifier,
    'RandomForest': RandomForestClassifier,
    'RidgeClassifier': RidgeClassifier,
    'Lasso': MultiClassModel(Lasso)
}

evaluation_models = [
    (SVC(), 'SupportVectorMachine'),
    (RandomForestClassifier, 'RandomForest')
]

evaluation_scoring = {
    "accuracy": make_scorer(accuracy_score),
    "macro_recall": make_scorer(recall_score, average='macro'),
    "macro_precision": make_scorer(precision_score, average='macro', zero_division=False),
    "micro_f1": make_scorer(f1_score, average='micro'),
    "macro_f1": make_scorer(f1_score, average='macro'),
    "roc_auc": make_scorer(one_vs_all_roc_auc)
}
