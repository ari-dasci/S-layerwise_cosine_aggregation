import scipy.stats as st
import torch

from typing import List


def group_by_layer(weights: List[List[torch.Tensor]]):
    return [
        torch.stack([weights[i][j] for i in range(len(weights))])
        for j in range(len(weights[0]))
    ]


def mean_sd_for_grouped_weights(grouped_weights: List[List[torch.Tensor]]):
    return [torch.mean(layer, dim=0) for layer in grouped_weights], [
        torch.std(layer, dim=0) for layer in grouped_weights
    ]


# Byzantine attack based on the paper "Little is enough"
def little_is_enough_attack(
    weights: List[List[torch.Tensor]], clients_per_round: int, poisoned_clients: int
):
    weights_by_layer = group_by_layer(weights)
    means, sds = mean_sd_for_grouped_weights(weights_by_layer)

    s = int(clients_per_round / 2 + 1) - poisoned_clients
    z = st.norm.ppf((clients_per_round - s) / clients_per_round)

    malicious_weights = [mean + z * sd for mean, sd in zip(means, sds)]
    return [malicious_weights for _ in range(clients_per_round)]
