import numpy as np

from results.loader import ResultsLoader


class ExecutionTimesAggregator:
    def __init__(
        self,
        results_loader: ResultsLoader,
    ):
        self._results_loader = results_loader

    def aggregated_execution_times(self):
        df = self._results_loader.load_all()

        max_features_per_alg_dataset = \
            df[['name', 'dataset_name', 'num_selected']] \
            .groupby(['name', 'dataset_name']) \
            .agg(np.max) \
            .reset_index()

        filtered_by_max = \
            max_features_per_alg_dataset \
            .merge(df, on=['name', 'dataset_name', 'num_selected'], how='inner')

        return \
            filtered_by_max[['name', 'dataset_name', 'num_selected', 'processing_time']] \
            .groupby(['name', 'dataset_name', 'num_selected']) \
            .agg('mean').reset_index()
