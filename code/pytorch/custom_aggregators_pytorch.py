"""
Custom aggregation functions for strategy implementations in PyTorch.
These functions operate on lists of NumPy arrays (model weights).
"""
import numpy as np
from functools import reduce
from typing import List, Tuple
from flwr.common import NDArrays

def aggregate_WgtAvg(results_with_id: List[Tuple[NDArrays, int, str]]) -> NDArrays:
    """
    Computes the weighted average of model parameters.
    Each tuple contains (weights, num_examples, client_id).
    """
    results_sorted = sorted(results_with_id, key=lambda x: x[2])
    num_examples_total = sum(num_examples for (_, num_examples, _) in results_sorted)
    weighted_weights = [
        [layer * num_examples for layer in weights]
        for (weights, num_examples, _) in results_sorted
    ]
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime

def aggregate_FedAvg(results_with_id: List[Tuple[NDArrays, int, str]]) -> NDArrays:
    """
    Computes the unweighted average of model parameters.
    Each tuple contains (weights, num_examples, client_id).
    """
    results_sorted = sorted(results_with_id, key=lambda x: x[2])
    all_weights = [weights for (weights, _num_examples, _client_id) in results_sorted]
    num_clients = len(all_weights)
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_clients
        for layer_updates in zip(*all_weights)
    ]
    return weights_prime
