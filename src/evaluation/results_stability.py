import json
from copy import deepcopy

import numpy as np
import pandas as pd

from results.loader import ResultsLoader
from .util import rank_from_weights, keep_top_k
from .stability import stability_for_sets, stability_for_ranks, stability_for_weights


class ResultsStability:
    def __init__(
        self,
        results_loader: ResultsLoader,
        evaluate_at=[5, 10, 20, 50, 100, 200]
    ):
        self._results_loader = results_loader
        self._evaluate_at = evaluate_at

    def _summarize_algorithm_stability(self, stability):
        fields = {
            'executions': np.sum,
            'jaccard': np.mean,
            'hamming': np.mean,
            'dice': np.mean,
            'ochiai': np.mean,
            'kuncheva': np.mean,
            'pog': np.mean,
            'spearman': np.mean,
            'pearson': np.mean,
        }

        return stability.drop(['dataset', 'feats'], axis=1).groupby(['name', 'selected']).agg(fields)

    def algorithms_stability(self, sampling='none', evaluate_at_max_features=False):
        df = self._results_loader.load_by_sampling(sampling)
        return self.stability_for_results(df, evaluate_at_max_features)

    def summarized_algorithms_stability(
        self,
        sampling='none',
        return_complete=False,
        evaluate_at_max_features=False
    ):
        complete_stability = self.algorithms_stability(sampling, evaluate_at_max_features)
        summarized_stability = self._summarize_algorithm_stability(complete_stability)
        if return_complete:
            return summarized_stability, complete_stability
        else:
            return summarized_stability

    def _stability_for_result(self, df, evaluate_at_max_features=True):
        values = np.stack(deepcopy(df['values']).apply(json.loads).values)

        name = deepcopy(df['name'].iloc[0])
        dataset_name = deepcopy(df['dataset_name'].iloc[0])
        num_selected = deepcopy(df['num_selected'].iloc[0])
        num_features = deepcopy(df['num_features'].iloc[0])
        result_type = deepcopy(df['result_type'].iloc[0])

        result_model = {
            'name': name,
            'dataset': dataset_name,
            'executions': len(values),
            'feats': num_features,
            'selected': num_selected,
        }

        evaluate_at_k = [k for k in set(self._evaluate_at) if k <= num_selected]
        if not evaluate_at_max_features and num_selected != num_features:
            evaluate_at_k += [num_selected]
        elif evaluate_at_max_features and num_selected == num_features:
            evaluate_at_k += [num_selected]

        ranks = values
        weights = values

        if result_type == 'weights':
            ranks = np.apply_along_axis(rank_from_weights, 1, np.stack(values))

        if result_type in ['weights', 'rank']:
            all_results = []
            for k in evaluate_at_k:
                results = deepcopy(result_model)

                rank_at_k = ranks[:, :k]

                results = {
                    **results,
                    **stability_for_ranks(rank_at_k, num_features),
                    'selected': k
                }

                if result_type == 'weights':
                    if k != num_features:
                        weights_at_k = [keep_top_k(w, k, set_others_to=0) for w in weights]
                    else:
                        weights_at_k = weights

                    weights_result = stability_for_weights(weights_at_k)
                    results.update(weights_result)

                all_results.append(results)

            return pd.DataFrame(all_results)

        if result_type == 'subset':
            subset_results = {**result_model, **stability_for_sets(values, num_features)}
            return pd.DataFrame([subset_results])

    def stability_for_results(self, df, evaluate_at_max_features=True):
        return df \
            .groupby(['name', 'dataset_name', 'num_selected']) \
            .apply(self._stability_for_result, evaluate_at_max_features=evaluate_at_max_features) \
            .reset_index(drop=True)
