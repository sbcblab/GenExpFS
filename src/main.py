from multiprocessing import Pool, Lock, cpu_count
import warnings

from data.dataloader import DataLoader
from data.shared_datasets import SharedDatasets
from data.results import ResultsWritter
from task.runner import TaskRunner

from util.shared_resources import SharedResources
from util.command_line import get_args
from util.task_creation_helper import get_presets


warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    args = get_args()
    results_path = args.results_path

    dataloader = DataLoader('datasets', normalize=True)

    datasets = SharedDatasets(dataloader)

    datasets_paths = {
        # Xor Dataset
        'xor_500samples_50features': 'xor/xor_500samples_50features.csv',

        # Cumida Datasets
        'Liver_GSE22405': 'cumida/Liver_GSE22405.csv',
        'Prostate_GSE6919_U95C': 'cumida/Prostate_GSE6919_U95C.csv',
        'Breast_GSE70947': 'cumida/Breast_GSE70947.csv',
        'Renal_GSE53757': 'cumida/Renal_GSE53757.csv',
        'Colorectal_GSE44861': 'cumida/Colorectal_GSE44861.csv',

        # Synthetic Datasets
        'synth_100samples_5000features_50informative':
            'synthetic/synth_100samples_5000features_50informative.csv',
        'synth_100samples_5000features_50informative_50redundant':
            'synthetic/synth_100samples_5000features_50informative_50redundant.csv',
        'synth_100samples_5000features_50informative_50redundant_50repeated':
            'synthetic/synth_100samples_5000features_50informative_50redundant_50repeated.csv',
        'synth_100samples_5000features_50informative_50repeated':
            'synthetic/synth_100samples_5000features_50informative_50repeated.csv',
        'synth_200samples_5000features_50informative':
            'synthetic/synth_200samples_5000features_50informative.csv',
        'synth_200samples_5000features_50informative_50redundant':
            'synthetic/synth_200samples_5000features_50informative_50redundant.csv',
        'synth_200samples_5000features_50informative_50redundant_50repeated':
            'synthetic/synth_200samples_5000features_50informative_50redundant_50repeated.csv',
        'synth_200samples_5000features_50informative_50repeated':
            'synthetic/synth_200samples_5000features_50informative_50repeated.csv',
    }

    datasets.add_datasets(datasets_paths)

    shared_resources = {
        "datasets": datasets,
        "lock": Lock()
    }

    results_writter = ResultsWritter(results_path)
    task_runner = TaskRunner(results_writter)

    tasks = get_presets()

    with Pool(cpu_count(), SharedResources.set_resources, [shared_resources]) as pool:
        pool.map(task_runner.run, tasks)


if __name__ == '__main__':
    main()
