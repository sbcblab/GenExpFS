import argparse


def get_args(arguments=None):
    parser = argparse.ArgumentParser(description='GenExpsFS')

    parser.add_argument(
        '-r',
        '--results_path',
        default='results',
        help='Path to where selection results will be saved.'
    )

    parser.add_argument(
        '-v',
        '--verbose',
        action='count'
    )

    args = parser.parse_args(arguments)
    return args
