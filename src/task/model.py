from feature_selectors.base_models.base_selector import BaseSelector


class Task():
    def __init__(
        self,
        name: str,
        feature_selector: BaseSelector,
        dataset_name: str,
        sampling: str = 'none'
    ):
        self.name = name
        self.feature_selector = feature_selector
        self.dataset_name = dataset_name
        self.sampling = sampling
