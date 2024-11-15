# Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by the LBANN Research Team (B. Van Essen, et al.) listed in
# the CONTRIBUTORS file. See the top-level LICENSE file for details.
#
# LLNL-CODE-697807.
# All rights reserved.
#
# This file is part of LBANN: Livermore Big Artificial Neural Network
# Toolkit. For details, see http://software.llnl.gov/LBANN or
# https://github.com/LBANN and https://github.com/LLNL/LBANN.
#
# SPDX-License-Identifier: (Apache-2.0)
import json
import os
from concurrent.futures import ThreadPoolExecutor

import torch
import tqdm
from safetensors import safe_open
from torch import nn


def load_checkpoint(
    model: nn.Module,
    folder: str,
    tp_rank: int,
    tp_world_size: int,
    device: torch.device,
    io_threads: int = 4,
):
    """
    Loads a safetensors checkpoint from a folder with tensor and pipeline
    parallelism support.

    :param model: The model to load the checkpoint into.
    :param folder: The folder where the checkpoint is stored as safetensors
                   files. NOTE: A ``model.safetensors.index.json`` file must
                   exist in the folder.
    :param tp_rank: The rank of the tensor parallelism group.
    :param tp_world_size: The world size of the tensor parallelism group.
    :param device: The device to load the model on.
    :note: Operates in-place on the model.
    """
    with open(os.path.join(folder, "model.safetensors.index.json"), "r") as fp:
        index = json.load(fp)
    # If a partial model is loaded (e.g., in pipeline parallelism), read only
    # the files that have parameters
    files = _intersect_tensors(model, index["weight_map"])

    # Get current device
    dev = "cpu"  # Faster than ``str(device)``

    params = {k: v for k, v in model.named_parameters()}

    with ThreadPoolExecutor(max_workers=io_threads) as executor:
        futures = []
        for file in sorted(files):
            filepath = os.path.join(folder, file)
            futures.append(
                executor.submit(
                    _load_file, filepath, params, dev, tp_rank, tp_world_size
                )
            )
        for future in tqdm.tqdm(
            futures, desc="Loading necessary safetensors files", total=len(futures)
        ):
            future.result()


def _intersect_tensors(model: nn.Module, available_tensors: dict[str, str]) -> set[str]:
    """
    Returns a set of files that parameters need to be loaded from.
    """
    result = set()
    for pname, _ in model.named_parameters():
        if pname in available_tensors:
            result.add(available_tensors[pname])
    return result


def _load_file(filepath, params, device, tp_rank, tp_world_size):
    with safe_open(filepath, framework="pt", device=device) as f:
        for key in sorted(f.keys()):
            if key in params:
                _load_tensor_fully_or_partially(f, key, params, tp_rank, tp_world_size)


def _load_tensor_fully_or_partially(
    f, key: str, params: dict[str, torch.nn.Parameter], tp_rank: int, tp_world_size: int
):
    """
    Loads a tensor from a safetensors file.

    :param f: The safetensors file.
    :param key: The key of the tensor to load.
    :param params: The parameters of the model.
    :param tp_rank: The rank of the tensor parallelism group.
    :param tp_world_size: The world size of the tensor parallelism group.
    """
    slc = f.get_slice(key)
    param = params[key]

    if tp_world_size > 1:  # Requires partial load
        # The following code would cause all the tensors to be loaded by tp_rank=0
        # and then broadcasted to the other ranks. We can go back to using this
        # function when PyTorch adds an optional root rank
        # slc_as_dtensor = distribute_tensor(slc[:], param.data.device_mesh, param.data.placements)
        # param.data[:] = slc_as_dtensor[:]
        # return

        shape = slc.get_shape()
        param_shape = param.data.shape
        diffs = [1 if (s != sts) else 0 for s, sts in zip(param_shape, shape)]
        if sum(diffs) == 0:  # No tensor parallelism
            param.data[:] = slc[:]
            return

        # Tensor parallelism (1D)
        if sum(diffs) > 1:
            raise ValueError("Only 1D parallelism is currently supported")
        tp_dim = next(i for i, d in enumerate(diffs) if d == 1)

        # Get the total size and compute slice offset
        chunk_size = shape[tp_dim] // tp_world_size
        chunk_offset = tp_rank * chunk_size

        # Prepare slice
        # Use parameter shape to account for uneven distribution across ranks
        ndslice = [slice(0, s, 1) for s in param_shape]
        ndslice[tp_dim] = slice(chunk_offset, chunk_offset + param_shape[tp_dim], 1)

        # Copy slice
        param.data[:] = slc[tuple(ndslice)]
    else:
        # Full load
        param.data[:] = slc[:]
