import math

import numpy as np
from numpy.random import random, randint, uniform, permutation
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import minmax_scale
from sklearn.svm import SVC

from feature_selectors.base_models.base_selector import BaseSelector, ResultType


def svd_f_score_fitness(X, y):
    eval_model = SVC()
    scoring = {"macro_f1": make_scorer(f1_score, average='macro')}
    cv = StratifiedKFold()
    results = cross_validate(eval_model, X, y, cv=cv, scoring=scoring)
    return results["test_macro_f1"].mean()


class GeneticAlgorithmFeatureSelector(BaseSelector):
    def __init__(
        self,
        n_features=50,
        num_individuals='auto',
        max_generations=100,
        num_elite=0.05,
        crossover_rate=0.5,
        mutation_rate=0.001,
        max_fitness=1.0,
        fitness_function=svd_f_score_fitness,
        verbose=0
    ):
        super().__init__(ResultType.SUBSET, n_features)
        self._n = n_features
        self._num_individuals = num_individuals
        self._max_generations = max_generations
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._max_fitness = max_fitness
        self._fitness_function = fitness_function
        self._verbose = verbose
        self._is_fitted = False

        def check_type(val, name, types=[int]):
            if type(val) not in types:
                raise ValueError(f"{name} must be one of {types}")

        check_type(num_elite, 'num_elite', [float, int])
        self._num_elite = num_elite

        check_type(num_individuals, 'num_individuals', [int, str])
        if type(num_individuals) is str and not num_individuals == 'auto':
            raise ValueError("Unknown value for num_individuals, must be `int` or 'auto'")

    def _auto_population(self, n_features):
        feature_pool = permutation(n_features)
        remaining = n_features % self._n
        if remaining > 0:
            padding = np.random.choice(feature_pool[:-remaining], self._n - remaining, replace=False)
            feature_pool = np.concatenate((feature_pool, padding))

        num_individuals = len(feature_pool) / self._n

        return np.split(feature_pool, num_individuals)

    def _initial_population(self, n_features):
        """Generate initial population as diverse as possible, if
        self._num_individuals is set to 'auto', then population
        size is gonna be the minimum needed so that each gene
        happens at least once in first generation"""

        if self._num_individuals == 'auto':
            return self._auto_population(n_features)

        num_ind_to_all = math.ceil(n_features / self._n)

        if self._num_individuals < num_ind_to_all:
            feature_pool = permutation(n_features)
            cut_point = self._num_individuals * self._n
            return np.split(feature_pool[:cut_point], self._num_individuals)

        else:
            needed_pops = math.ceil(self._num_individuals / math.ceil(n_features / self._n))
            population = np.concatenate([self._auto_population(n_features) for x in range(needed_pops)])
            return population[:self._num_individuals]

    def _mutate(self, individual, n_features):
        for i in range(self._n):
            if random() < self._mutation_rate:
                value = randint(0, n_features)
                loop = True
                while(loop):
                    if value not in individual:
                        loop = False
                        individual[i] = value
                    value = randint(0, n_features)
        return individual

    def _crossover(self, parent1, parent2):
        diff_p1 = np.setdiff1d(parent1, parent2)
        num_diff = len(diff_p1)

        if num_diff == 0:
            return parent1, parent2

        diff_p2 = np.setdiff1d(parent2, parent1)

        diff_p1 = permutation(diff_p1)
        diff_p2 = permutation(diff_p2)

        for i in range(num_diff):
            if random() < self._crossover_rate:
                diff_p1[i], diff_p2[i] = diff_p2[i], diff_p1[i]

        if num_diff < self._n:
            common = np.intersect1d(parent1, parent2)
            return np.concatenate((common, diff_p1)), np.concatenate((common, diff_p2))
        else:
            return diff_p1, diff_p2

    def _selection(self, population, fitness):
        scaled_fitness = minmax_scale(fitness)
        fitness_sum = scaled_fitness.sum()
        selected = []
        for i in range(self._num_to_select):
            current_sum = uniform(0, fitness_sum)
            i = 0
            while (current_sum < fitness_sum):
                current_sum += scaled_fitness[i]
                i += 1
            selected.append(np.copy(population[i - 1]))

        return selected

    def fit(self, X, y):
        self.check_already_fitted()
        self._X = X
        n_samples, n_features = X.shape
        self._total_features = n_features

        population = np.array(self._initial_population(n_features))
        fitness = np.array([self._fitness_function(X[:, x], y) for x in population])

        num_individuals = len(population)
        self.num_individuals_ = num_individuals

        num_elite = int(self._num_elite * num_individuals) if type(self._num_elite) is float else self._num_elite

        num_to_select = num_individuals - num_elite
        self._num_to_select = num_to_select + (num_to_select % 2)

        for gen in range(self._max_generations):
            fitness_index = np.argsort(fitness)[::-1]
            elite_index = fitness_index[:num_elite]
            elite_fitness = fitness[elite_index]
            elite = population[elite_index]

            best_fitness = fitness[fitness_index[0]]

            if self._verbose > 0:
                print(f"Generation [{gen + 1}/{self._max_generations}]: Best fitness: {fitness[fitness_index[0]]}")

            if best_fitness >= self._max_fitness:
                print("Early stop, found optimal solution!")
                break

            new_population = self._selection(population, fitness)

            offspring_pairs = [
                self._crossover(new_population[i], new_population[i + 1])
                for i in range(0, self._num_to_select, 2)
            ]

            offsprings = [offspring for osp in offspring_pairs for offspring in osp]

            [self._mutate(individual, n_features) for individual in offsprings]

            offsprings_fitness = [self._fitness_function(X[:, x], y) for x in offsprings]

            population = np.concatenate((elite, offsprings))
            fitness = np.concatenate((elite_fitness, offsprings_fitness))

        best_index = np.argsort(fitness)[-1]
        self.last_generation_ = population
        self.last_generation_fitness_ = fitness
        self._selected_fitness = fitness[best_index]
        self._selected = population[best_index]
        self._support_mask = np.zeros(self._total_features)
        self._support_mask[self._selected] = True
        self._is_fitted = True

        return self
