import json

import numpy as np
import pandas as pd

from util.dict import flatten_dict
from results.loader import ResultsLoader


DEFAULT_COLOR = '\033[39m'
CYAN_COLOR = '\033[36m'
GREEN_COLOR = '\033[32m'
YELLOW_COLOR = '\033[33m'
WHITE_COLOR = '\033[37m'


class ResultsScorer:
    def __init__(
        self,
        results_loader: ResultsLoader,
        datasets,
        selection_scorer,
        evaluate_at=[5, 10, 20, 50, 100, 200],
        verbose=1
    ):
        self._results_loader = results_loader
        self._datasets = datasets
        self._selection_scorer = selection_scorer
        self._evaluate_at = evaluate_at
        self._verbose = verbose

    def score_all(self):
        subset_results = self._results_loader.load_by_result_type('subset')
        rank_results = self._results_loader.load_by_result_type('rank')
        weights_results = self._results_loader.load_by_result_type('weights')

        subset_scores = self.evaluate_subsets(subset_results)
        rank_scores = self.evaluate_ordered(rank_results)
        weights_scores = self.evaluate_ordered(weights_results)

        return pd.concat([subset_scores, rank_scores, weights_scores])

    def _dataset_from_result(self, result):
        dataset_name = result['dataset_name']
        X, y, _ = self._datasets.get_dataset(dataset_name).get()
        return X, y

    def _values_from_result(self, result):
        return json.loads(result['values'])

    def _weights_to_rank(self, weights):
        return [int(x) for x in np.argsort(weights)[::-1]]

    def _print(self, result):
        if self._verbose > 0:
            print(
                f"{YELLOW_COLOR}Evaluated results for {GREEN_COLOR}{result['name']}\n"
                f"{WHITE_COLOR}  type:{CYAN_COLOR} {result['result_type']}\n"
                f"{WHITE_COLOR}  dataset:{CYAN_COLOR} {result['dataset_name']}\n"
                f"{WHITE_COLOR}  features:{CYAN_COLOR} {result['num_selected']}{DEFAULT_COLOR}"
            )

    def evaluate_subsets(self, subset_results):
        def evaluate(result):
            selected = self._values_from_result(result)
            X, y = self._dataset_from_result(result)
            eval_results = self._selection_scorer.eval(X[:, selected], y)
            self._print(result)

            return pd.Series({**result, **flatten_dict(eval_results)})

        return subset_results.apply(evaluate, axis=1)

    def evaluate_ordered(self, results):
        def evaluate(result):
            result_type = result['result_type']
            values = self._values_from_result(result)

            if result_type == 'rank':
                rank = values
            elif result_type == 'weights':
                rank = self._weights_to_rank(values)
            else:
                raise Exception("Result type must be either `rank` or `weights`")

            X, y = self._dataset_from_result(result)

            results_data = []
            for k in self._evaluate_at:
                if k >= len(rank):
                    continue
                selected = rank[:k]
                eval_results = self._selection_scorer.eval(X[:, selected], y)
                results = {**result, **flatten_dict(eval_results)}
                results['values'] = json.dumps(selected)
                results['num_selected'] = k
                results_data.append(results)
                self._print(results)

            return pd.DataFrame(results_data)

        return pd.concat(results.apply(evaluate, axis=1).to_list()).reset_index(drop=True)
