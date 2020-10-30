import json
from copy import deepcopy
from multiprocessing import Pool

import numpy as np
import pandas as pd

from results.loader import ResultsLoader
from .util import rank_from_weights, keep_top_k
from .stability import stability_for_sets, stability_for_ranks, stability_for_weights


class ResultsStability:
    def __init__(
        self,
        results_loader: ResultsLoader,
        evaluate_at=[5, 10, 20, 50, 100, 200],
        verbose=1,
        n_workers=1
    ):
        self._results_loader = results_loader
        self._evaluate_at = evaluate_at
        self._verbose = verbose
        self._n_workers = n_workers

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

        return \
            stability.drop(['dataset', 'feats'], axis=1) \
            .groupby(['name', 'selected']) \
            .agg(fields).reset_index()

    def algorithms_stability(self, sampling=None, evaluate_at_all_features=False):
        if sampling is not None:
            df = self._results_loader.load_by_sampling(sampling)
        else:
            df = self._results_loader.load_all()

        return self.stability_for_results(df, evaluate_at_all_features)

    def summarized_algorithms_stability(
        self,
        sampling=None,
        return_complete=False,
        evaluate_at_all_features=False
    ):
        complete_stability = self.algorithms_stability(sampling, evaluate_at_all_features)
        summarized_stability = self._summarize_algorithm_stability(complete_stability)
        if return_complete:
            return summarized_stability, complete_stability
        else:
            return summarized_stability

    def _print_evaluating(self, name, dataset_name, num_selected, num_executions):
        if self._verbose > 0:
            print(
                f"Evaluating stability for {name}:\n"
                f"  dataset: {dataset_name}\n"
                f"  number of features selected: {num_selected}\n"
                f"  executions: {num_executions}"
            )

    def _stability_for_result(self, df, evaluate_at_all_features=True):
        values = np.stack(deepcopy(df['values']).apply(json.loads).values)

        name = deepcopy(df['name'].iloc[0])
        dataset_name = deepcopy(df['dataset_name'].iloc[0])
        num_selected = deepcopy(df['num_selected'].iloc[0])
        num_features = deepcopy(df['num_features'].iloc[0])
        result_type = deepcopy(df['result_type'].iloc[0])

        num_executions = len(values)

        result_model = {
            'name': name,
            'dataset': dataset_name,
            'executions': num_executions,
            'feats': num_features,
            'selected': num_selected,
        }

        evaluate_at_k = [k for k in self._evaluate_at if k <= num_selected]
        if num_selected not in evaluate_at_k and evaluate_at_all_features and num_selected == num_features:
            evaluate_at_k += [num_selected]

        ranks = values
        weights = values

        if result_type == 'weights':
            ranks = np.apply_along_axis(rank_from_weights, 1, np.stack(values))

        if result_type in ['weights', 'rank']:
            all_results = []
            for k in evaluate_at_k:
                results = deepcopy(result_model)

                self._print_evaluating(name, dataset_name, k,  num_executions)

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
            self._print_evaluating(name, dataset_name, num_selected, num_executions)
            subset_results = {**result_model, **stability_for_sets(values, num_features)}
            return pd.DataFrame([subset_results])

    def stability_for_results(self, df, evaluate_at_all_features=True):
        if self._verbose > 0:
            print("Starting stability analysis.")

        grouped = df.groupby(['name', 'dataset_name', 'num_selected'])
        groups = [g for i, g in grouped]

        if self._verbose > 0:
            print(f"Grouped results in {len(groups)} groups.")

        if self._n_workers > 1:
            with Pool(self._n_workers) as pool:
                eval_at_max = [evaluate_at_all_features for _ in groups]

                stabilities = pool.starmap(
                    self._stability_for_result,
                    zip(groups, eval_at_max)
                )

        else:
            stabilities = []
            for g in groups:
                result = self._stability_for_result(g, evaluate_at_all_features)
                stabilities.append(result)

        return pd.concat(stabilities)
