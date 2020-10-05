import numpy as np
from numpy.random import random, randint, uniform, permutation

from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import minmax_scale
from sklearn.svm import SVC


def svd_f_score_fitness(X, y):
    eval_model = SVC()
    scoring = {"macro_f1": make_scorer(f1_score, average='macro')}
    cv = StratifiedKFold()
    results = cross_validate(eval_model, X, y, cv=cv, scoring=scoring)
    return results["test_macro_f1"].mean()


class GeneticAlgorithmFeatureSelector:
    def __init__(
        self,
        n_features=50,
        num_individuals=50,
        max_generations=100,
        num_elite=0.1,
        cross_over_rate=0.5,
        mutation_rate=0.01,
        max_fitness=1.0,
        fitness_function=svd_f_score_fitness,
        verbose=0
    ):
        self._n = n_features
        self._num_individuals = num_individuals
        self._max_generations = max_generations
        self._mutation_rate = mutation_rate
        self._cross_over_rate = cross_over_rate
        self._max_fitness = max_fitness
        self._fitness_function = fitness_function
        self._verbose = verbose

        if isinstance(num_elite, float) and num_elite < 1 and num_elite >= 0:
            self._num_elite = int(num_individuals * num_elite)
        elif isinstance(num_elite, int) and num_elite < num_individuals:
            self._num_elite = num_elite
        else:
            raise ValueError("num_elite must be a `float` between 0.0 and 1.0 or an `int` < num_individuals")

        num_to_select = num_individuals - self._num_elite
        self._num_to_select = num_to_select + (num_to_select % 2)

    def _make_random(self, n_features):
        individual = []
        while len(individual) < self._n:
            feat = randint(0, n_features)
            if feat not in individual:
                individual.append(feat)
        return np.array(individual)

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

    def _uniform_crossover(self, parent1, parent2):
        p1 = permutation(parent1)
        p2 = permutation(parent2)

        diff_p1 = np.setdiff1d(p1, p2)
        num_diff = len(diff_p1)

        if num_diff == 0:
            return p1, p2

        diff_p2 = np.setdiff1d(p2, p1)

        for i in range(num_diff):
            if random() < self._cross_over_rate:
                tmp = diff_p1[i]
                diff_p1[i] = diff_p2[i]
                diff_p2[i] = tmp

        common = np.intersect1d(p1, p2)
        if len(common) > 0:
            return np.concatenate((common, diff_p1)), np.concatenate((common, diff_p2))
        else:
            return diff_p1, diff_p2

    def _one_point_crossover(self, parent1, parent2):
        p1 = permutation(parent1)
        p2 = permutation(parent2)

        diff_p1 = np.setdiff1d(p1, p2)
        num_diff = len(diff_p1)

        upper_bound = int(num_diff * 0.8)
        if upper_bound == 0:
            return None

        lower_bound = int(num_diff * 0.2)
        cut = randint(lower_bound, upper_bound)

        diff_p2 = np.setdiff1d(p2, p1)
        child1 = np.concatenate((diff_p1[:cut], diff_p2[cut:]))
        child2 = np.concatenate((diff_p2[:cut], diff_p1[cut:]))

        common = np.intersect1d(p1, p2)
        if len(common) > 0:
            return np.concatenate((common, child1)), np.concatenate((common, child2))
        else:
            return child1, child2

    def _cross_over(self, parent1, parent2):
        children1 = self._one_point_crossover(parent1, parent2)
        children2 = self._uniform_crossover(parent1, parent2)
        if children1:
            return children1 + children2
        else:
            return children2

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
        n_samples, n_features = X.shape

        population = np.array([self._make_random(n_features) for i in range(self._n)])
        fitness = np.array([self._fitness_function(X[:, x], y) for x in population])

        for gen in range(self._max_generations):

            fitness_index = np.argsort(fitness)[::-1]
            elite_index = fitness_index[:self._num_elite]
            elite_fitness = fitness[elite_index]
            elite = population[elite_index]

            best_fitness = fitness[fitness_index[0]]

            if self._verbose > 0:
                print(f"Generation [{gen + 1}/{self._max_generations}]: Best fitness: {fitness[fitness_index[0]]}")

            if best_fitness >= self._max_fitness:
                print("Early stop, found optimal solution!")
                break

            new_population = self._selection(population, fitness)

            child_pairs = [
                self._cross_over(new_population[i], new_population[i + 1])
                for i in range(0, self._num_to_select, 2)
            ]

            children = [child for cp in child_pairs for child in cp]

            [self._mutate(individual, n_features) for individual in children]

            children_fitness = [self._fitness_function(X[:, x], y) for x in children]

            population = np.concatenate((elite, children))
            fitness = np.concatenate((elite_fitness, children_fitness))

        best_index = np.argsort(fitness)[-1]
        self.selected_features_ = population[best_index]
