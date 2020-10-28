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
        scores = []

        try:
            subset_results = self._results_loader.load_by_result_type('subset')
            subset_scores = self.evaluate_subsets(subset_results)
            scores.append(subset_scores)
        except Exception as e:
            print(f"Could not load subset results, reson: {e}")

        try:
            rank_results = self._results_loader.load_by_result_type('rank')
            rank_scores = self.evaluate_ordered(rank_results)
            scores.append(rank_scores)
        except Exception as e:
            print(f"Could not load rank results, reson: {e}")

        try:
            weights_results = self._results_loader.load_by_result_type('weights')
            weights_scores = self.evaluate_ordered(weights_results)
            scores.append(weights_scores)
        except Exception as e:
            print(f"Could not load weighted results, reson: {e}")

        if len(scores) == 0:
            raise Exception("Could generate any scoring for given results.")

        return pd.concat(scores)

    def _summarized_scores(self, scores):
        fields = {
            'SupportVectorMachine_macro_f1': np.mean,
            'DecisionTree_macro_f1': np.mean,
            'RandomForest_macro_f1': np.mean,
            'NaiveBayes_macro_f1': np.mean,
            'ZeroR_macro_f1': np.mean,
        }

        return scores.groupby(['name', 'selected']).agg(fields).reset_index()

    def summarized_score_all(self, return_complete=False):
        complete_scoring = self.score_all()
        summarized_scoring = self._summarized_scores(complete_scoring)
        if return_complete:
            return summarized_scoring, complete_scoring
        else:
            return summarized_scoring

    def _dataset_from_result(self, result):
        dataset_name = result['dataset_name']
        X, y, _ = self._datasets.get_dataset(dataset_name).get()
        return X, y

    def _values_from_result(self, result):
        return json.loads(result['values'])

    def _weights_to_rank(self, weights):
        return [int(x) for x in np.argsort(weights)[::-1]]

    def _get_result_model(self, result):
        return {
            'name': result['name'],
            'processing_time': result['processing_time'],
            'dataset': result['dataset_name'],
            'features': result['num_features'],
            'selected': result['num_selected'],
            'sampling': result['sampling'],
        }

    def _print(self, result):
        if self._verbose > 1:
            print(
                f"{YELLOW_COLOR}Evaluated results for {GREEN_COLOR}{result['name']}\n"
                f"{WHITE_COLOR}  dataset:{CYAN_COLOR} {result['dataset']}\n"
                f"{WHITE_COLOR}  features:{CYAN_COLOR} {result['features']}{DEFAULT_COLOR}\n"
                f"{WHITE_COLOR}  selected:{CYAN_COLOR} {result['selected']}{DEFAULT_COLOR}"
            )
        elif self._verbose > 0:
            print(
                f"Evaluated results for {result['name']}\n"
                f"  dataset: {result['dataset']}\n"
                f"  selected: {result['selected']}\n"
                f"  features: {result['features']}"
            )

    def evaluate_subsets(self, subset_results):
        def evaluate(result):
            selected = self._values_from_result(result)
            X, y = self._dataset_from_result(result)
            eval_results = self._selection_scorer.eval(X[:, selected], y)

            result_model = self._get_result_model(result)
            self._print(result_model)

            return pd.Series({**result_model, **flatten_dict(eval_results)})

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

            result_model = self._get_result_model(result)

            results_data = []
            for k in self._evaluate_at:
                if k >= len(rank):
                    continue
                selected = rank[:k]
                eval_results = self._selection_scorer.eval(X[:, selected], y)
                results = {**result_model, **flatten_dict(eval_results)}
                results['num_selected'] = k
                results_data.append(results)
                self._print(results)

            return pd.DataFrame(results_data)

        return pd.concat(results.apply(evaluate, axis=1).to_list()).reset_index(drop=True)
