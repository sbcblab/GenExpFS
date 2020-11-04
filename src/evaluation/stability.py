from itertools import combinations

import numpy as np

from evaluation.metrics import (
    jaccard_score,
    set_normalized_hamming_distance,
    dice_coefficient,
    kuncheva_index,
)

from evaluation.statistics import (
    spearmans_correlation_partial_ranked_list,
    canberra_distance_partial_ranked_list,
    pearsons_correlation_no_zeros
)


def averaged_stability(selections, metric, as_type=lambda x: x, *args, **kwargs):
    if len(selections) == 1:
        return metric(as_type(selections[0]), as_type(selections[0]), *args, **kwargs)

    return np.mean([
        metric(as_type(s1), as_type(s2), *args, **kwargs)
        for s1, s2
        in combinations(selections, 2)
    ])


def stability_for_sets(selections, num_features):
    return {
        'jaccard': averaged_stability(selections, jaccard_score, set),
        'hamming': averaged_stability(selections, set_normalized_hamming_distance, set),
        'dice': averaged_stability(selections, dice_coefficient, set),
        'kuncheva': averaged_stability(selections, kuncheva_index, set, num_features)
    }


def stability_for_ranks(selections, num_features):
    return {
        **stability_for_sets(selections, num_features),
        'spearman': averaged_stability(selections, spearmans_correlation_partial_ranked_list, np.array),
        'canberra': averaged_stability(selections, canberra_distance_partial_ranked_list, np.array),
    }


def stability_for_weights(selections):
    return {'pearson': averaged_stability(selections, pearsons_correlation_no_zeros, np.array)}
