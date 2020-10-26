import json
import os
from itertools import product, chain


from task.model import Task
from feature_selectors import (
    DecisionTreeFeatureSelector,
    KruskalWallisFeatureSelector,
    LassoFeatureSelector,
    LinearSVMFeatureSelector,
    LRForwardFeatureSelector,
    LRGAFeatureSelector,
    MRMRFeatureSelector,
    MutualInformationFeatureSelector,
    RandomForestFeatureSelector,
    ReliefFFeatureSelector,
    RidgeClassifierFeatureSelector,
    SVMForwardFeatureSelector,
    SVMGAFeatureSelector,
    MRMRGAFeatureSelector
)


feature_selectors = {
    "MRMR": MRMRFeatureSelector,
    "SVMFowardSelection": SVMForwardFeatureSelector,
    "LRFowardSelection": LRForwardFeatureSelector,
    "MutualInformationFilter": MutualInformationFeatureSelector,
    "KruskallWallisFilter": KruskalWallisFeatureSelector,
    "ReliefFFeatureSelector": ReliefFFeatureSelector,
    "Lasso": LassoFeatureSelector,
    "RidgeClassifier": RidgeClassifierFeatureSelector,
    "LinearSVM": LinearSVMFeatureSelector,
    "LogisticRegressionGeneticAlgorithm": LRGAFeatureSelector,
    "SVMGeneticAlgorithm": SVMGAFeatureSelector,
    "DecisionTree": DecisionTreeFeatureSelector,
    "RandomForest": RandomForestFeatureSelector,
    "MRMRGeneticAlgorithm": MRMRGAFeatureSelector,
}


def load_preset(path):
    dirname = os.path.dirname(__file__)
    presets_path = os.path.join(dirname, 'presets')
    file_path = os.path.join(presets_path, path)

    with open(file_path) as f:
        return json.load(f)


def config_to_tasks(config):
    for entry in config:
        for dataset, alg in product(entry['datasets'], entry['algs']):
            for params in alg['params']:
                for _ in range(alg['runs']):
                    name = alg['name']
                    yield Task(name, feature_selectors[name](*params), dataset, False)
                for _ in range(alg['bootstrap_runs']):
                    name = alg['name']
                    yield Task(name, feature_selectors[name](*params), dataset, True)


def test_presets():
    test_config = load_preset('test.json')
    return config_to_tasks(test_config)


def default_presets(runs=100):
    det_config = load_preset('default/deterministic.json')
    bs_det_config = load_preset('default/bootstrap_deterministic.json')
    stoc_config = load_preset('default/stochastic.json')
    bs_stoc_config = load_preset('default/bootstrap_stochastic.json')

    tasks = [config_to_tasks(det_config)]
    for r in range(runs):
        tasks.append(config_to_tasks(bs_det_config))
        tasks.append(config_to_tasks(stoc_config))
        tasks.append(config_to_tasks(bs_stoc_config))

    return chain(*tasks)
