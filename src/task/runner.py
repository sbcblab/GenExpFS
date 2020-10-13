import json
from time import time

from data.sampling import bootstrap
from data.results import Result
from feature_selectors.base_models.base_selector import ResultType
from .model import Task
from util.shared_resources import SharedResources


class TaskRunner():
    def __init__(self, results_writter):
        self._results_writter = results_writter

    def run(self, task: Task):
        shared_resources = SharedResources.get()
        datasets = shared_resources['datasets']
        lock = shared_resources['lock']

        dataset = datasets.get_dataset(task.dataset_name)
        X = dataset.get_instances()
        y = dataset.get_classes()

        if task.bootstrap:
            X, y = bootstrap(X, y)

        fs = task.feature_selector

        start = time()
        fs.fit(X, y)
        time_spent = time() - start

        result_type = fs.get_result_type()
        if result_type is ResultType.WEIGHTS:
            values = list(fs.get_weights())
        elif result_type is ResultType.RANK:
            values = list(fs.get_rank())
        else:
            values = list(fs.get_selected())

        num_selected = task.feature_selector._n_features

        result = Result(
            name=task.name,
            processing_time=time_spent,
            task_type=task.kind.value,
            dataset_path=dataset.path,
            num_features=dataset.get_instances_shape()[1],
            num_selected=num_selected if num_selected else -1,
            sampling='boostrap' if task.bootstrap else 'none',
            result_type=result_type.value,
            values=json.dumps(values)
        )

        lock.acquire()
        try:
            self._results_writter.write(result)
        finally:
            lock.release()
