# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.


import numpy as np
import torch


def extend(tensor: torch.Tensor, dims: int) -> torch.Tensor:
    """Extend tensor to match dimensions 'dims'."""
    return tensor[(...,) + (None,) * dims]


def extend_to(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Extend tensor1 to have same number of dims as tensor2."""
    return extend(tensor1, len(tensor2.shape) - len(tensor1.shape))


def repeat_to(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Repeat tensor1 to match the shape of tensor2."""
    tensor1 = extend_to(tensor1, tensor2)
    ones = torch.ones(tensor2.shape[:-1] + (1,), device=tensor1.device)
    return tensor1 * ones


def replicate_params_for_chains(params: dict, n_chains: int) -> dict:
    """
    Replicate each problem's parameters for a group of processors/chains.

    The output ordering is [problem_0 chain_0, ..., problem_0 chain_n,
    problem_1 chain_0, ...], so each contiguous chain group receives the
    same problem instance.
    """
    if n_chains < 1:
        raise ValueError("n_chains must be >= 1")
    if n_chains == 1:
        return params
    return {k: v.repeat_interleave(n_chains, dim=0) for k, v in params.items()}


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array."""
    return tensor.detach().cpu().numpy()
