from enum import Enum

from feature_selectors.base_models.base_selector import BaseSelector


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
        dataset_name: str,
        bootstrap: bool = False
    ):
        self.name = name
        self.feature_selector = feature_selector
        self.kind = kind
        self.dataset_name = dataset_name
        self.bootstrap = bootstrap
