from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import Lasso, RidgeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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

evaluation_models = [
    (SVC(), 'SupportVectorMachine'),
    (RandomForestClassifier, 'RandomForest')
]

embedded_selection = {
    (DecisionTreeClassifier, 'DecisionTree'),
    (RandomForestClassifier, 'RandomForest'),
    (RidgeClassifier, 'RidgeClassifier'),
    (MultiClassModel(Lasso), 'Lasso')
}
