import json
from time import time

from data.sampling import bootstrap
from results.model import Result
from feature_selectors.base_models import ResultType
from .model import Task
from util.shared_resources import SharedResources

DEFAULT_COLOR = '\033[39m'
RED_COLOR = "\033[31m"
CYAN_COLOR = '\033[36m'
GREEN_COLOR = '\033[32m'


class TaskRunner():
    def __init__(self, results_writter):
        self._results_writter = results_writter

    def run(self, task: Task):
        print(f"{CYAN_COLOR}Starting task {task.name} for dataset: {task.dataset_name}{DEFAULT_COLOR}")
        try:
            shared_resources = SharedResources.get()
            datasets = shared_resources['datasets']
            lock = shared_resources['lock']
        except Exception as e:
            print(f"{RED_COLOR}Failed to get shared resources! Reason: {e}{DEFAULT_COLOR}")
            return

        try:
            dataset = datasets.get_dataset(task.dataset_name)
            X = dataset.get_instances()
            y = dataset.get_classes()

            if task.bootstrap:
                X, y = bootstrap(X, y)
        except Exception as e:
            print(
                f"{RED_COLOR}Failed to load dataset `{task.dataset_name}` "
                f"for task `{task.name}`! Reason: {e}{DEFAULT_COLOR}"
            )
            return

        fs = task.feature_selector

        try:
            start = time()
            fs.fit(X, y)
            time_spent = time() - start
        except Exception as e:
            print(
                f"{RED_COLOR}Failed to fit model for dataset `{task.dataset_name}` "
                f"and task `{task.name}`! Reason: {e}{DEFAULT_COLOR}"
            )
            raise e

        try:
            if fs.result_type is ResultType.WEIGHTS:
                values = list(fs.get_weights())
            elif fs.result_type is ResultType.RANK:
                values = [int(v) for v in fs.get_rank()]
            else:
                values = [int(v) for v in fs.get_selected()]

            num_selected = task.feature_selector._n_features

            result = Result(
                name=task.name,
                processing_time=time_spent,
                dataset_name=dataset.name,
                num_features=dataset.get_instances_shape()[1],
                num_selected=num_selected if num_selected else -1,
                sampling='bootstrap' if task.bootstrap else 'none',
                result_type=fs.result_type.value,
                values=json.dumps(values)
            )

            lock.acquire()
            try:
                self._results_writter.write(result)
                print(f"{GREEN_COLOR}Task {task.name} done! written results for {task.dataset_name}{DEFAULT_COLOR}")
            finally:
                lock.release()
        except Exception as e:
            print(
                f"{RED_COLOR}Failed to save results task `{task.name}` and dataset "
                f"`{task.dataset_name}`! Reason: {e}{DEFAULT_COLOR}"
            )
