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
        for dataset, algorithm in product(entry['datasets'], entry['algorithms']):
            for params in algorithm['params']:
                for _ in range(algorithm['runs']):
                    name = algorithm['name']
                    yield Task(name, feature_selectors[name](*params), dataset, False)
                for _ in range(algorithm['bootstrap_runs']):
                    name = algorithm['name']
                    yield Task(name, feature_selectors[name](*params), dataset, True)


def print_preset(name, preset, verbose, runs):
    if verbose > 0:
        print(f"Loaded configs from preset {name}:")
        for config in preset:

            description = config.get('description')
            config_runs = 1 if config.get('run_once') else runs

            alg_runs = [
                (alg['name'], alg['runs'])
                for alg in config.get('algorithms')
                if alg['runs'] > 0
            ]

            bootstrap_alg_runs = [
                (alg['name'], alg['bootstrap_runs'])
                for alg in config.get('algorithms')
                if alg['bootstrap_runs'] > 0
            ]

            if description is not None:
                print(f"\t{description}")
                print("\t\tDatasets:")
                [print(f'\t\t\t{d}') for d in config['datasets']]
                print(f"\t\tPreset repeat times: {config_runs}")
                print("\t\tAlgorithm runs:")
                [print(f'\t\t\t{a}: {r} per dataset') for a, r in alg_runs]
                print("\t\tAlgorithm Bootstrap runs:")
                [print(f'\t\t\t{a}: {r} per dataset') for a, r in bootstrap_alg_runs]


def tasks_from_presets(preset_names, runs=1, verbose=0):
    tasks = []
    for name in preset_names:
        try:
            _name = name if name.endswith('.json') else f'{name}.json'
            preset = load_preset(_name)

            print_preset(name, preset, verbose, runs)

            run_once_configs = [config for config in preset if config.get('run_once')]
            tasks.append(config_to_tasks(run_once_configs))

            run_n_configs = [config for config in preset if not config.get('run_once')]
            for _ in range(runs):
                tasks.append(config_to_tasks(run_n_configs))
        except Exception as e:
            print(f"Could not load preset {name} because: {e}")

    final_tasks = list(chain(*tasks))

    if verbose > 0:
        print(f"Generated {len(final_tasks)} tasks for {preset_names} presets!")

    return final_tasks
