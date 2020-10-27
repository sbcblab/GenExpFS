from multiprocessing import Pool, Lock
import os
import warnings

from sklearn.exceptions import ConvergenceWarning

from data.dataloader import DataLoader
from data.shared_datasets import SharedDatasets
from results.writter import ResultsWritter
from results.loader import ResultsLoader
from task.runner import TaskRunner
from util.shared_resources import SharedResources
from util.command_line import get_args
from util.task_creation_helper import tasks_from_presets

from evaluation.results_scorer import ResultsScorer
from evaluation.results_stability import ResultsStability
from evaluation.selection import SelectionScorer


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
ConvergenceWarning('ignore')


def main():
    args = get_args()

    mode = args.mode

    if mode in ['all', 'select']:
        num_workers = args.workers
        presets = args.presets
        presets_runs = args.presets_runs

    results_path = args.results_path

    if mode in ['all', 'select', 'scoring']:
        datasets_path = args.datasets_path

    selection_filename = args.selection_filename

    if mode in ['all', 'scoring']:
        scoring_filename = args.scoring_filename

    if mode in ['all', 'stability']:
        stability_filename = args.stability_filename
        data_stability_filename = args.data_stability_filename

    # if mode in ['all', 'times']:
    #     times_filename = args.times_filename

    if mode in ['all', 'select', 'scoring']:
        dataloader = DataLoader(datasets_path, normalize=True)
        datasets = SharedDatasets(dataloader)

        datasets_paths = {
            # Xor Dataset
            'xor_500samples_50features': 'xor/xor_500samples_50features.csv',

            # Cumida Datasets
            # 'Liver_GSE22405': 'cumida/Liver_GSE22405.csv',
            # 'Prostate_GSE6919_U95C': 'cumida/Prostate_GSE6919_U95C.csv',
            # 'Breast_GSE70947': 'cumida/Breast_GSE70947.csv',
            # 'Renal_GSE53757': 'cumida/Renal_GSE53757.csv',
            # 'Colorectal_GSE44861': 'cumida/Colorectal_GSE44861.csv',

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

    if mode in ['all', 'select']:
        task_runner = TaskRunner(results_writter, selection_filename)
        tasks = tasks_from_presets(presets, presets_runs)

        with Pool(num_workers, SharedResources.set_resources, [shared_resources]) as pool:
            pool.map(task_runner.run, tasks)

    results_loader = ResultsLoader(os.path.join(results_path, selection_filename))

    if mode in ['all', 'scoring']:
        selection_scorer = SelectionScorer()
        scorer = ResultsScorer(results_loader, datasets, selection_scorer)
        scoring_results = scorer.score_all()

        results_writter.write_dataframe(scoring_results, scoring_filename)

    if mode in ['all', 'stability']:
        stability_evaluator = ResultsStability(results_loader)

        try:
            alg_stab_sum, alg_stab = stability_evaluator.summarized_algorithms_stability(
                sampling='none', return_complete=True
            )

            results_writter.write_dataframe(alg_stab, stability_filename)
            results_writter.write_dataframe(alg_stab_sum, f'complete-{stability_filename}')
        except Exception as e:
            print(f"Could not run results stability evaluation. Reason: {e}")

        try:
            alg_data_stab_sum, alg_data_stab = stability_evaluator.summarized_algorithms_stability(
                sampling='bootstrap', return_complete=True
            )

            results_writter.write_dataframe(alg_data_stab, data_stability_filename)
            results_writter.write_dataframe(alg_data_stab_sum, f'complete-{data_stability_filename}')
        except Exception as e:
            print(f"Could not run data stability evaluation. Reason: {e}")


if __name__ == '__main__':
    main()
