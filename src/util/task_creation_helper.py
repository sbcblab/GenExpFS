from itertools import product, chain

import numpy as np

from task.model import Task
from feature_selectors import (
    DecisionTreeFeatureSelector,
    KruskalWallisFeatureSelector,
    LassoFeatureSelector,
    LRForwardFeatureSelector,
    LRGAFeatureSelector,
    MRMRFeatureSelector,
    MutualInformationFeatureSelector,
    RandomForestFeatureSelector,
    ReliefF,
    RidgeClassifierFeatureSelector,
    SVMForwardFeatureSelector,
    SVMGAFeatureSelector,
    MRMRGAFeatureSelector
)


class _TaskDescriptor:
    def __init__(self, feature_selector, name, args=[], kwargs={}, n_features=[None]):
        self.feature_selector = feature_selector
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.n_features = n_features


def interpolate_tasks(runs, datasets=[], bootstrap=False, task_descs=[]):
    for r in range(runs):
        task_dataset = np.random.permutation(list(product(task_descs, datasets)))
        for td, dataset in task_dataset:
            for n_feats in td.n_features:
                feature_selector = td.feature_selector(n_features=n_feats, *td.args, **td.kwargs)
                yield Task(td.name, feature_selector, dataset, bootstrap)


def mrmr(n_features):
    return [
        _TaskDescriptor(MRMRFeatureSelector, "MRMR", n_features=n_features)
    ]


def sequential_forward_search(n_features):
    return [
        _TaskDescriptor(SVMForwardFeatureSelector, "SVMFowardSelection", n_features=n_features),
        _TaskDescriptor(LRForwardFeatureSelector, "LRFowardSelection", n_features=n_features)
    ]


def deterministic_algs():
    return [
        _TaskDescriptor(MutualInformationFeatureSelector, "MutualInformationFilter"),
        _TaskDescriptor(KruskalWallisFeatureSelector, "KruskallWallisFilter"),
        _TaskDescriptor(ReliefF, "ReliefF"),
        _TaskDescriptor(LassoFeatureSelector, "Lasso"),
        _TaskDescriptor(RidgeClassifierFeatureSelector, "RidgeClassifier")
    ]


def genetic_algorithms(n_features):
    return [
        _TaskDescriptor(LRGAFeatureSelector, "LogisticRegressionGeneticAlgorithm", n_features=n_features),
        _TaskDescriptor(SVMGAFeatureSelector, "SVMGeneticAlgorithm", n_features=n_features),
    ]


def tree_and_forest_algs():
    return [
        _TaskDescriptor(DecisionTreeFeatureSelector, "DecisionTree"),
        _TaskDescriptor(RandomForestFeatureSelector, "RandomForest")
    ]


def mrmr_ga_task(n_features, n_features_mrmr=1000):
    return _TaskDescriptor(
        MRMRGAFeatureSelector,
        "MRMRGeneticAlgorithm",
        kwargs={"n_features_mrmr": n_features_mrmr},
        n_features=n_features
    )


def get_test_presets():
    task_descriptors = deterministic_algs()
    task_descriptors += genetic_algorithms([20])
    task_descriptors += tree_and_forest_algs()

    task_descriptors += mrmr([20])
    task_descriptors += sequential_forward_search([20])
    task_descriptors += [mrmr_ga_task([20], 30)]
    return interpolate_tasks(1, ['xor_500samples50feats'], False, task_descriptors)


def generate_tasks(
    runs, datasets, ga_feats, ga_runs, mrmr_feats, mrmr_runs, sfs_feats, sfs_runs, mmrmr_ga_feats, mrmr_ga_runs
):
    mrmr_descriptors = mrmr(mrmr_feats)
    sfs_descriptors = sequential_forward_search(sfs_feats)
    det_task_descriptors = deterministic_algs()

    det_tasks = interpolate_tasks(1, datasets, False, mrmr_descriptors + sfs_descriptors + det_task_descriptors)

    mrmr_bootstrap = interpolate_tasks(mrmr_runs, datasets, True, mrmr_descriptors)
    sfs_bootstrap = interpolate_tasks(sfs_runs, datasets, True, sfs_descriptors)
    det_tasks_bootstrap = interpolate_tasks(runs, datasets, True, det_tasks)

    ga_descriptors = genetic_algorithms(ga_feats)
    tree_forest_descriptors = tree_and_forest_algs()
    mrmr_ga_descriptors = [mrmr_ga_task(*x) for x in mmrmr_ga_feats]

    ga_tasks = interpolate_tasks(runs, datasets, False, ga_descriptors)
    tree_forest_tasks = interpolate_tasks(runs, datasets, False, tree_forest_descriptors)
    mrmr_ga_tasks = interpolate_tasks(runs, datasets, False, mrmr_ga_descriptors)
    return chain(
        det_tasks, det_tasks_bootstrap, sfs_bootstrap, mrmr_bootstrap, ga_tasks, tree_forest_tasks, mrmr_ga_tasks
    )


def get_presets(test=False):
    if test:
        return get_test_presets()

    xor_presets = generate_tasks(
        100,
        ['xor_500samples50feats'],
        [5, 10, 20],
        100,
        [None],
        50,
        [None],
        20,
        [([5], 10), ([10], 20), ([20], 30)],
        50
    )

    cumida_presets = generate_tasks(
        50,
        ['Liver_GSE22405', 'Prostate_GSE6919_U95C', 'Breast_GSE70947', 'Renal_GSE53757', 'Colorectal_GSE44861'],
        [5, 10, 20, 50, 100, 200],
        30,
        [200],
        20,
        [200],
        10,
        [([5, 10, 20, 50, 100, 200],)],
        30
    )

    tasks = list(chain(xor_presets, cumida_presets))
    np.random.shuffle(tasks)
    return tasks
