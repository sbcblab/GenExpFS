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
    MRMRGAFeatureSelector,
    MutualInformationFeatureSelector,
    RandomForestFeatureSelector,
    ReliefFFeatureSelector,
    ReliefFGAFeatureSelector,
    RidgeClassifierFeatureSelector,
    SVMForwardFeatureSelector,
    SVMGAFeatureSelector,
    SVMRFE
)


feature_selectors = {
    "DecisionTree": DecisionTreeFeatureSelector,
    "KruskallWallisFilter": KruskalWallisFeatureSelector,
    "Lasso": LassoFeatureSelector,
    "LinearSVM": LinearSVMFeatureSelector,
    "LogisticRegressionGeneticAlgorithm": LRGAFeatureSelector,
    "LRFowardSelection": LRForwardFeatureSelector,
    "MRMR": MRMRFeatureSelector,
    "MRMRGeneticAlgorithm": MRMRGAFeatureSelector,
    "MutualInformationFilter": MutualInformationFeatureSelector,
    "RandomForest": RandomForestFeatureSelector,
    "ReliefFFeatureSelector": ReliefFFeatureSelector,
    "ReliefFGeneticAlgorithm": ReliefFGAFeatureSelector,
    "RidgeClassifier": RidgeClassifierFeatureSelector,
    "SVMFowardSelection": SVMForwardFeatureSelector,
    "SVMGeneticAlgorithm": SVMGAFeatureSelector,
    "SVMRFE": SVMRFE,
}


def load_preset(path):
    dirname = os.path.dirname(__file__)
    presets_path = os.path.join(dirname, 'presets')
    file_path = os.path.join(presets_path, path)

    with open(file_path, "r") as f:
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


def tasks_from_presets(preset_names, runs=1):
    tasks = []
    for name in preset_names:
        try:
            preset = load_preset(name)
            for _ in range(runs):
                tasks.append(config_to_tasks(preset))
        except Exception as e:
            if name == 'default':
                tasks.append(default_tasks(runs))
            else:
                print(f"Could not load preset {name}, reason: {e}")

    return chain(*tasks)


def test_presets():
    test_config = load_preset('test.json')
    return config_to_tasks(test_config)


def default_tasks(runs=100):
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
