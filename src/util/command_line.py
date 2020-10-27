import argparse
from multiprocessing import cpu_count
from time import time


current_timestamp = int(time())


def add_num_workers(parser):
    parser.add_argument(
        '-w',
        '--workers',
        default=cpu_count(),
        help='Number of workers that are gonna be spawned to run selection tasks. [default: cpu count]',
        type=int
    )


def add_results_path(parser):
    parser.add_argument(
        '-r',
        '--results-path',
        default='results',
        help='Path to where selection results will be saved.',
        type=str
    )


def add_datasets_path(parser):
    parser.add_argument(
        '-d',
        '--datasets_path',
        default='datasets',
        help='Path to datasets.',
        type=str
    )


def add_presets(parser):
    parser.add_argument(
        '-p',
        '--presets',
        nargs='*',
        default=['test'],
        help='List of presets to be used in feature selection tasks.'
    )


def add_presets_runs(parser):
    parser.add_argument(
        '-n',
        '--presets-runs',
        default=1,
        help='Number of times presets will be run.',
        type=int
    )


def add_selection_filename(parser):
    parser.add_argument(
        '--selection-filename',
        default=f'selection-{current_timestamp}',
        help='File name of file where selection results will be saved.',
        type=str
    )


def add_scoring_filename(parser):
    parser.add_argument(
        '--scoring-filename',
        default=f'scoring-{current_timestamp}',
        help='File name of file where scoring results will be saved.',
        type=str
    )


def add_stability_filename(parser):
    parser.add_argument(
        '--stability-filename',
        default=f'stability-{current_timestamp}',
        help='File name of file where stability results will be saved.',
        type=str
    )


def add_data_stability_filename(parser):
    parser.add_argument(
        '--data-stability-filename',
        default=f'data-stability-{current_timestamp}',
        help='File name of file where data stability results will be saved.',
        type=str
    )


def add_times_filename(parser):
    parser.add_argument(
        '--times-filename',
        default=f'times-{current_timestamp}',
        help='File name of file where execution times will be saved.',
        type=str
    )


def add_verbosity(parser):
    parser.add_argument(
        '-v',
        '--verbose',
        action='count',
        default=0
    )


def get_args(arguments=None):
    parser = argparse.ArgumentParser(description='GenExpFS')
    help_str = 'Framework to run feature selections algorithms and evaluate performance and stability on their results'
    subparsers = parser.add_subparsers(dest='mode', help=help_str)
    subparsers.required = True

    # Main Parser (Run all)
    all_parser = subparsers.add_parser('all', help='Run entire selection and evaluation pipeline.')
    add_num_workers(all_parser)
    add_results_path(all_parser)
    add_datasets_path(all_parser)
    add_presets(all_parser)
    add_selection_filename(all_parser)
    add_scoring_filename(all_parser)
    add_stability_filename(all_parser)
    add_data_stability_filename(all_parser)
    add_times_filename(all_parser)
    add_verbosity(all_parser)

    # Feature Selection Command
    feature_selection_parser = subparsers.add_parser('select', help='Run feature selection tasks.')
    add_num_workers(feature_selection_parser)
    add_results_path(feature_selection_parser)
    add_datasets_path(feature_selection_parser)
    add_presets(feature_selection_parser)
    add_selection_filename(feature_selection_parser)
    add_verbosity(feature_selection_parser)

    # Scoring Command
    scoring_parser = subparsers.add_parser('scoring', help='Run selection scoring tasks.')
    add_results_path(scoring_parser)
    add_datasets_path(scoring_parser)
    add_selection_filename(scoring_parser)
    add_scoring_filename(scoring_parser)
    add_verbosity(scoring_parser)

    # Stability Evaluation Command
    stability_parser = subparsers.add_parser('stability', help='Run stability evaluation tasks.')
    add_results_path(stability_parser)
    add_selection_filename(stability_parser)
    add_stability_filename(stability_parser)
    add_data_stability_filename(stability_parser)
    add_verbosity(stability_parser)

    # Execution time Command
    times_parser = subparsers.add_parser('times', help='Run execution time evaluation tasks.')
    add_results_path(times_parser)
    add_selection_filename(times_parser)
    add_times_filename(times_parser)
    add_verbosity(times_parser)

    args = parser.parse_args(arguments)
    return args
