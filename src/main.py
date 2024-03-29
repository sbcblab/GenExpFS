from multiprocessing import Pool, Lock
import os
import warnings

from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

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
from evaluation.results_execution_time import ExecutionTimesAggregator
from evaluation.selection import SelectionScorer


warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)
ConvergenceWarning('ignore')
UndefinedMetricWarning('ignore')


def main():
    args = get_args()

    mode = args.mode
    verbose = args.verbose

    if mode in ['all', 'select']:
        presets = args.presets
        presets_runs = args.presets_runs

    if mode in ['all', 'select', 'stability', 'determinism']:
        num_workers = args.workers

    results_path = args.results_path

    if mode in ['all', 'select', 'scoring']:
        datasets_path = args.datasets_path

    selection_filename = args.selection_filename

    if mode in ['all', 'scoring']:
        scoring_filename = args.scoring_filename

    if mode in ['all', 'determinism']:
        determinism_filename = args.determinism_filename

    if mode in ['all', 'stability']:
        stability_filename = args.stability_filename

    if mode in ['all', 'times']:
        times_filename = args.times_filename

    if mode in ['all', 'select', 'scoring']:
        dataloader = DataLoader(datasets_path, normalize=True)
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

    if mode in ['all', 'select']:
        task_runner = TaskRunner(results_writter, selection_filename, verbose=verbose)
        tasks = tasks_from_presets(presets, presets_runs, verbose=verbose)

        with Pool(num_workers, SharedResources.set_resources, [shared_resources]) as pool:
            pool.map(task_runner.run, tasks)

    selection_filename = selection_filename if selection_filename.endswith('.csv') else f'{selection_filename}.csv'
    results_loader = ResultsLoader(os.path.join(results_path, selection_filename))

    if mode in ['all', 'scoring']:
        selection_scorer = SelectionScorer()
        scorer = ResultsScorer(results_loader, datasets, selection_scorer, verbose=verbose)
        summarized_scoring, scoring = scorer.summarized_score_all(return_complete=True)

        results_writter.write_dataframe(summarized_scoring, scoring_filename)
        results_writter.write_dataframe(scoring, f'{scoring_filename}-complete')

    if mode in ['all', 'stability']:
        stability_evaluator = ResultsStability(results_loader, verbose=verbose, n_workers=num_workers)

        try:
            alg_stab_sum, alg_stab = stability_evaluator.summarized_algorithms_stability(
                sampling='bootstrap', return_complete=True
            )

            results_writter.write_dataframe(alg_stab_sum, f'{stability_filename}-bootstrap')
            results_writter.write_dataframe(alg_stab, f'{stability_filename}-bootstrap-complete')
        except Exception as e:
            print(f"Could not run data stability evaluation. Reason: {e}")

        try:
            alg_stab_sum, alg_stab = stability_evaluator.summarized_algorithms_stability(
                sampling='percent90', return_complete=True
            )

            results_writter.write_dataframe(alg_stab_sum, f'{stability_filename}-90perecent')
            results_writter.write_dataframe(alg_stab, f'{stability_filename}-90percent-complete')
        except Exception as e:
            print(f"Could not run data stability evaluation. Reason: {e}")

    if mode in ['all', 'determinism']:
        stability_evaluator = ResultsStability(results_loader, verbose=verbose, n_workers=num_workers)

        try:
            alg_det_sum, alg_det = stability_evaluator.summarized_algorithms_stability(
                sampling='none', return_complete=True
            )

            results_writter.write_dataframe(alg_det_sum, determinism_filename)
            results_writter.write_dataframe(alg_det, f'{determinism_filename}-complete')
        except Exception as e:
            print(f"Could not run results determinism evaluation. Reason: {e}")

    if mode in ['all', 'times']:
        times_aggregator = ExecutionTimesAggregator(results_loader)
        exec_times = times_aggregator.aggregated_execution_times()
        results_writter.write_dataframe(exec_times, times_filename)


if __name__ == '__main__':
    main()
