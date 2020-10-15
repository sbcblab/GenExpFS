import json
from time import time

from data.sampling import bootstrap
from data.results import Result
from feature_selectors.base_models import ResultType
from .model import Task
from util.shared_resources import SharedResources


class TaskRunner():
    def __init__(self, results_writter):
        self._results_writter = results_writter

    def run(self, task: Task):
        print(f"Starting task {task.name} for dataset: {task.dataset_name}")
        try:
            shared_resources = SharedResources.get()
            datasets = shared_resources['datasets']
            lock = shared_resources['lock']
        except Exception as e:
            print(f"Failed to get shared resources! Reason: {e}")
            return

        try:
            dataset = datasets.get_dataset(task.dataset_name)
            X = dataset.get_instances()
            y = dataset.get_classes()

            if task.bootstrap:
                X, y = bootstrap(X, y)
        except Exception as e:
            print(f"Failed to load dataset `{task.dataset_name}` for task `{task.name}`! Reason: {e}")
            return

        fs = task.feature_selector

        try:
            start = time()
            fs.fit(X, y)
            time_spent = time() - start
        except Exception as e:
            print(f"Failed to fit model for dataset `{task.dataset_name}` and task `{task.name}`! Reason: {e}")
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
                dataset_path=dataset.path,
                num_features=dataset.get_instances_shape()[1],
                num_selected=num_selected if num_selected else -1,
                sampling='boostrap' if task.bootstrap else 'none',
                result_type=fs.result_type.value,
                values=json.dumps(values)
            )

            lock.acquire()
            try:
                self._results_writter.write(result)
                print(f"Task {task.name} done! written results for {task.dataset_name}")
            finally:
                lock.release()
        except Exception as e:
            print(f"Failed to save results task `{task.name}` and dataset `{task.dataset_name}`! Reason: {e}")
