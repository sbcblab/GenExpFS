from enum import Enum

from feature_selectors.base_selector import BaseSelector, ResultType


class SelectorKind(Enum):
    FILTER = "filter"
    WRAPPER = "wrapper"
    EMBEDDED = "embedded"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"


class Task():
    def __init__(
        self,
        name: str,
        feature_selector: BaseSelector,
        kind: SelectorKind,
        dataset: str,
        num_features: int,
        result_type: ResultType,
        bootstrap: bool = False
    ):
        self.name = name
        self.feature_selector = feature_selector
        self.dataset = dataset
        self.num_features = num_features
        self.result_type = result_type
        self.bootstrap = bootstrap
