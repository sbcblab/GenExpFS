from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import Lasso, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from multi_class_model import MultiClassModel
from select_k_significant_best import SelectKSignificantBest
from forward_feature_selector import ForwardFeatureSelector
from statistics import kruskal_wallis

datasets = [
    'breastCancer.csv',
    'xor_500samples50feats.csv'
]

filtering_strategies = {
    'SelectKBeast': SelectKBest,
    'SelectKSignificantBest': SelectKSignificantBest
}

filtering_functions = {
    'Anova': f_classif,
    'MultualInformation': mutual_info_classif,
    'KruskalWallis': kruskal_wallis
}

embedded_selection = {
    'DecisionTree': DecisionTreeClassifier(),
    'RidgeClassifier': RidgeClassifier(),
    'Lasso': MultiClassModel(Lasso)
}

wrapper_selection = {
    'ForwardFeatureSelector': ForwardFeatureSelector
}

ensemble_selection = {
    'RandomForest': RandomForestClassifier()
}
