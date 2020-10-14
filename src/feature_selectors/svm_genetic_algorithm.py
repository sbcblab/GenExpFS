from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.svm import SVC

from feature_selectors.base_models.genetic_algorithm import GeneticAlgorithmFeatureSelector


def svd_f_score_fitness(X, y):
    eval_model = SVC()
    scoring = {"macro_f1": make_scorer(f1_score, average='macro')}
    cv = StratifiedKFold(n_splits=5)
    results = cross_validate(eval_model, X, y, cv=cv, scoring=scoring)
    return results["test_macro_f1"].mean()


class SVMGAFeatureSelector(GeneticAlgorithmFeatureSelector):
    def __init__(
        self,
        n_features=50,
        num_individuals='auto',
        max_generations=100,
        num_elite=0.05,
        crossover_rate=0.5,
        mutation_rate=0.001,
        verbose=0
    ):
        super().__init__(
            n_features=n_features,
            num_individuals=num_individuals,
            max_generations=max_generations,
            num_elite=num_elite,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            max_fitness=1.0,
            fitness_function=svd_f_score_fitness,
            verbose=verbose,
        )
