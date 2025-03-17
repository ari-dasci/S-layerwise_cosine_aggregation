import re
from functools import reduce
from typing import Dict, List

import tensorly as tl
import torch
from flex.model import FlexModel
from flex.pool import aggregate_weights, collect_clients_weights
from flexclash.pool import bulyan, multikrum


@aggregate_weights
def layerwise_krum(weights: List[List[tl.tensor]], f: int = 1):
    return [
        multikrum.__wrapped__([[weights[j][i]] for j in range(len(weights))], m=1, f=f)[
            0
        ]
        for i in range(len(weights[0]))
    ]


@aggregate_weights
def layerwise_bulyan(weights: List[List[tl.tensor]], f: int = 1, m: int = 5):
    return [
        bulyan.__wrapped__([[weights[j][i]] for j in range(len(weights))], m=m, f=f)[0]
        for i in range(len(weights[0]))
    ]


def _krum_cosine_similarity(
    list_of_updates: List[torch.Tensor], f: int = 1, k: int = 1
) -> torch.Tensor:
    """
    Implements the Krum operator using cosine similarity in PyTorch.

    Args:
        updates (torch.Tensor): Tensor of shape (n_clients, update_size).
        f (int): Maximum number of Byzantine clients to tolerate.
        k (int): The ammount of updates to select

    Returns:
        torch.Tensor: The k indexes of the selected updates of the Krum operator.
    """
    updates = torch.stack(list_of_updates)
    n_clients = updates.size(0)
    norm_updates = updates / updates.norm(dim=1, keepdim=True)
    similarities = torch.mm(norm_updates, norm_updates.t())
    distances = 1 - similarities
    distances.fill_diagonal_(float("inf"))
    sorted_distances, _ = distances.sort(dim=1)
    m = n_clients - f - 2
    scores = sorted_distances[:, :m].sum(dim=1)
    min_score_index = torch.topk(scores, k, largest=False).indices
    return min_score_index


@aggregate_weights
def krum_cosine_similarity(list_of_weights: List[List[torch.Tensor]], f: int = 1):
    flattened_weights = [
        torch.cat([torch.flatten(param) for param in model_weights])
        for model_weights in list_of_weights
    ]

    selected_update = list_of_weights[
        _krum_cosine_similarity(flattened_weights, f).item()
    ]

    return selected_update


@aggregate_weights
def krum_cosine_similarity_layerwise(
    list_of_weights: List[List[torch.Tensor]], f: int = 1
):
    selected_update = [
        list_of_weights[
            _krum_cosine_similarity(
                [
                    torch.flatten(list_of_weights[j][i])
                    for j in range(len(list_of_weights))
                ],
                f,
            ).item()
        ][i]
        for i in range(len(list_of_weights[0]))
    ]

    param_shapes = [param.shape for param in list_of_weights[0]]
    selected_weights = []
    for param, shape in zip(selected_update, param_shapes):
        assert type(param) == torch.Tensor, f"param is not a tensor: {param}"
        param = param.view(shape)
        selected_weights.append(param)

    return selected_weights


def _bulyan_cosine_similarity(
    list_of_updates: List[torch.Tensor], f: int = 1, m: int = 5
) -> torch.Tensor:
    if m < 2 * f + 2:
        raise ValueError("m must be greater than 2*f+2")

    flattened_updates = [tensor.view(-1) for tensor in list_of_updates]
    k = m
    selected_updates_indexes = _krum_cosine_similarity(flattened_updates, f, k)
    selected_updates = [list_of_updates[i] for i in selected_updates_indexes]

    # Stack the selected updates
    stacked_updates = torch.stack(
        selected_updates
    )  # Shape: (num_selected, update_size)

    # Compute the Bulyan aggregation
    sorted_updates, _ = torch.sort(stacked_updates, dim=0)
    trimmed_updates = sorted_updates[f:-f]
    bulyan_update = trimmed_updates.mean(dim=0)

    return bulyan_update


@aggregate_weights
def bulyan_cosine_similarity(
    list_of_weights: List[List[torch.Tensor]], f: int = 1, m: int = 5
) -> List[torch.Tensor]:
    flattened_weights = [
        torch.cat([param.view(-1) for param in model_weights])
        for model_weights in list_of_weights
    ]
    selected_update = _bulyan_cosine_similarity(flattened_weights, f, m)

    new_update = []
    offset = 0
    for weights in list_of_weights[0]:
        size = weights.numel()
        new_update.append(selected_update[offset : offset + size].view(weights.shape))
        offset += size

    return new_update


@aggregate_weights
def bulyan_cosine_similarity_layerwise(
    list_of_updates: List[List[torch.Tensor]], f: int = 1, m: int = 5
) -> List[torch.Tensor]:
    selected_updates = [
        _bulyan_cosine_similarity([weights[i] for weights in list_of_updates], f, m)
        for i in range(len(list_of_updates[0]))
    ]

    return selected_updates


def clip_by_norm(aggregator_func):
    @aggregate_weights
    def _func(list_of_weights: List[List[torch.Tensor]], *args, **kwargs):
        norms_by_layer = [
            [torch.norm(param[i]) for param in list_of_weights]
            for i in range(len(list_of_weights[0]))
        ]

        medians = [torch.median(torch.tensor(norms)) for norms in norms_by_layer]
        selected_update = aggregator_func.__wrapped__(list_of_weights, *args, **kwargs)
        new_update = []
        for i, param in enumerate(selected_update):
            norm = torch.norm(param)
            if norm > medians[i]:
                param = param * (medians[i] / norm)
            new_update.append(param)

        return new_update

    return _func


def _geomed(list_of_weigths: List[torch.Tensor]):
    updates = torch.stack(list_of_weigths)
    distances = torch.cdist(updates, updates, p=2)
    dimensions_to_reduce = tuple(range(1, distances.dim()))
    sum_distances = distances.sum(dim=dimensions_to_reduce)
    min_index = sum_distances.argmin()
    return updates[min_index]


@aggregate_weights
def geomed(list_of_weights: List[List[torch.Tensor]]):
    updates_flattened = [
        torch.cat([param.view(-1) for param in model_weights])
        for model_weights in list_of_weights
    ]

    selected_update = _geomed(updates_flattened)

    return_value = []
    offset = 0
    for weights in list_of_weights[0]:
        size = weights.numel()
        return_value.append(selected_update[offset : offset + size].view(weights.shape))
        offset += size

    return return_value


@aggregate_weights
def layerwise_geomed(list_of_weights: List[List[torch.Tensor]]):
    selected_update = [
        _geomed([weights[i].view(-1) for weights in list_of_weights])
        for i in range(len(list_of_weights[0]))
    ]

    selected_update = [
        update.view(param.shape)
        for update, param in zip(selected_update, list_of_weights[0])
    ]
    return selected_update


def _cosine_geomed(list_of_weigths: List[torch.Tensor]):
    updates = torch.stack(list_of_weigths)
    updates = updates.view(updates.size(0), -1)
    norm_updates = updates / updates.norm(dim=1, keepdim=True)
    similarities = torch.mm(norm_updates, norm_updates.t())
    distances = 1 - similarities
    distances.fill_diagonal_(0)
    sum_distances = distances.sum(dim=1)
    min_index = sum_distances.argmin()
    return list_of_weigths[min_index]


@aggregate_weights
def cosine_geomed(list_of_weights: List[List[torch.Tensor]]):
    flattened_list = [
        torch.cat([param.view(-1) for param in model_weights])
        for model_weights in list_of_weights
    ]
    selected_update = _cosine_geomed(flattened_list)
    return_value = []
    offset = 0
    for weights in list_of_weights[0]:
        size = weights.numel()
        return_value.append(selected_update[offset : offset + size].view(weights.shape))
        offset += size

    return return_value


@aggregate_weights
def layerwise_cosine_geomed(list_of_weights: List[List[torch.Tensor]]):
    selected_update = [
        _cosine_geomed([weights[i] for weights in list_of_weights])
        for i in range(len(list_of_weights[0]))
    ]
    return selected_update
