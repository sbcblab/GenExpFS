import json
from time import time
import traceback

from data.sampling import bootstrap
from results.model import Result
from feature_selectors.base_models import ResultType
from .model import Task
from util.shared_resources import SharedResources


DEFAULT_COLOR = '\033[39m'
RED_COLOR = "\033[31m"
CYAN_COLOR = '\033[36m'
GREEN_COLOR = '\033[32m'
YELLOW_COLOR = '\033[33m'


class TaskRunner():
    def __init__(self, results_writter, output_file_name, verbose=1):
        self._results_writter = results_writter
        self._verbose = verbose
        self._output_file_name = output_file_name

    def _print_start(self, name, dataset_name):
        if self._verbose > 1:
            print(f"{CYAN_COLOR}Starting task {name} for dataset: {dataset_name}{DEFAULT_COLOR}")
        elif self._verbose > 0:
            print(f"[Start] Task {name} for dataset: {dataset_name}")

    def _print_end(self, name, dataset_name, time):
        if self._verbose > 1:
            print(
                f"{GREEN_COLOR}Task {name} done! written results for "
                f"{dataset_name}{YELLOW_COLOR} [{time:0.2f}s]{DEFAULT_COLOR}"
            )
        elif self._verbose > 0:
            print(f"[Done] task {name}! written results for {dataset_name} [{time:0.2f}s]")

    def _error(self, text):
        if self._verbose > 1:
            print(f"{RED_COLOR}{text}{DEFAULT_COLOR}")
            tb_string = traceback.format_exc(limit=5)
            print(f"{YELLOW_COLOR}{tb_string}{DEFAULT_COLOR}", end='')
        else:
            print(text)
            tb_string = traceback.format_exc(limit=5)
            print(tb_string, end='')

    def run(self, task: Task):
        self._print_start(task.name, task.dataset_name)
        try:
            shared_resources = SharedResources.get()
            datasets = shared_resources['datasets']
            lock = shared_resources['lock']
        except Exception:
            self._print("Failed to get shared resources!")
            return

        try:
            dataset = datasets.get_dataset(task.dataset_name)
            X = dataset.get_instances()
            y = dataset.get_classes()

            if task.bootstrap:
                X, y = bootstrap(X, y)
        except Exception:
            self._error(
                f"Failed to load dataset `{task.dataset_name}` for task `{task.name}`!"
            )
            return

        fs = task.feature_selector

        try:
            start = time()
            fs.fit(X, y)
            time_spent = time() - start
        except Exception:
            self._error(f"Failed to fit model for dataset `{task.dataset_name}` and task `{task.name}`!")
            return

        try:
            if fs.result_type is ResultType.WEIGHTS:
                values = list(fs.get_weights())
            elif fs.result_type is ResultType.RANK:
                values = [int(v) for v in fs.get_rank()]
            else:
                values = [int(v) for v in fs.get_selected()]

            num_selected = task.feature_selector._n_features
            num_features = dataset.get_instances_shape()[1]

            result = Result(
                name=task.name,
                processing_time=time_spent,
                dataset_name=dataset.name,
                num_features=num_features,
                num_selected=num_selected if num_selected else num_features,
                sampling='bootstrap' if task.bootstrap else 'none',
                result_type=fs.result_type.value,
                values=json.dumps(values)
            )

            lock.acquire()
            try:
                self._results_writter.write_result(result, self._output_file_name)
                self._print_end(task.name, task.dataset_name, time_spent)
            finally:
                lock.release()
        except Exception:
            self._error(f"Failed to save results task `{task.name}` and dataset `{task.dataset_name}`!")
