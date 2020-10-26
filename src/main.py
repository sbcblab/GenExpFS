from multiprocessing import Pool, Lock, cpu_count
import warnings

from sklearn.exceptions import ConvergenceWarning

from data.dataloader import DataLoader
from data.shared_datasets import SharedDatasets
from results.writter import ResultsWritter
from results.loader import ResultsLoader
from task.runner import TaskRunner
from util.shared_resources import SharedResources
from util.command_line import get_args
from util.task_creation_helper import test_presets

from evaluation.results_scorer import ResultsScorer
from evaluation.results_stability import ResultsStability
from evaluation.selection import SelectionScorer


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
ConvergenceWarning('ignore')


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
    }

    datasets.add_datasets(datasets_paths)

    shared_resources = {
        "datasets": datasets,
        "lock": Lock()
    }

    results_writter = ResultsWritter(results_path)
    task_runner = TaskRunner(results_writter)

    tasks = test_presets()

    with Pool(cpu_count(), SharedResources.set_resources, [shared_resources]) as pool:
        pool.map(task_runner.run, tasks)

    results_loader = ResultsLoader(results_path)

    selection_scorer = SelectionScorer()
    scorer = ResultsScorer(results_loader, datasets, selection_scorer)
    results = scorer.score_all()

    results_writter.write_dataframe(results, 'scored.csv')

    stability_evaluator = ResultsStability(results_loader)

    alg_stab_sum, alg_stab = stability_evaluator.summarized_algorithms_stability(
        sampling='none', return_complete=True
    )
    results_writter.write_dataframe(alg_stab, 'algorithm_stability.csv')
    results_writter.write_dataframe(alg_stab_sum, 'summarized_algorithm_stability.csv')

    alg_data_stab_sum, alg_data_stab = stability_evaluator.summarized_algorithms_stability(
        sampling='bootstrap', return_complete=True
    )

    results_writter.write_dataframe(alg_data_stab, 'algorithm_to_data_stability.csv')
    results_writter.write_dataframe(alg_data_stab_sum, 'summarized_algorithm_to_data_stability.csv')


if __name__ == '__main__':
    main()
