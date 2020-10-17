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

    dataloader = DataLoader('datasets')

    datasets = SharedDatasets(dataloader)

    datasets_paths = {
        'xor_500samples50feats': 'xor_500samples50feats.csv',
        # 'Liver_GSE22405': 'cumida_binary/Liver_GSE22405.csv',
        # 'Prostate_GSE6919_U95C': 'cumida_binary/Prostate_GSE6919_U95C.csv',
        # 'Breast_GSE70947': 'cumida_binary/Breast_GSE70947.csv',
        # 'Renal_GSE53757': 'cumida_binary/Renal_GSE53757.csv',
        # 'Colorectal_GSE44861': 'cumida_binary/Colorectal_GSE44861.csv'
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
