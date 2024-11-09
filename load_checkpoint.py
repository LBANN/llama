import json
import os
from safetensors import safe_open
import torch
from torch import nn
from torch.distributed.tensor import DTensor
import tqdm


def load_checkpoint(model: nn.Module, folder: str, tp_rank: int,
                    tp_world_size: int):
    """
    Loads a safetensors checkpoint from a folder with tensor and pipeline
    parallelism support.

    :param model: The model to load the checkpoint into.
    :param folder: The folder where the checkpoint is stored as safetensors
                   files. NOTE: A ``model.safetensors.index.json`` file must
                   exist in the folder.
    :param tp_rank: The rank of the tensor parallelism group.
    :param tp_world_size: The world size of the tensor parallelism group.
    :note: Operates in-place on the model.
    """
    with open(os.path.join(folder, 'model.safetensors.index.json'), 'r') as fp:
        index = json.load(fp)
    # If a partial model is loaded (e.g., in pipeline parallelism), read only
    # the files that have parameters
    files = _intersect_tensors(model, index['weight_map'])

    # Get current device
    dev = torch.device('cuda', torch.cuda.current_device())

    params = {k: v for k, v in model.named_parameters()}

    # Load the necessary files
    for file in tqdm.tqdm(files, desc='Loading necessary safetensors files'):
        filepath = os.path.join(folder, file)
        with safe_open(filepath, framework="pt", device=dev) as f:
            for key in f.keys():
                if key in params:
                    _load_tensor_fully_or_partially(f, key, params, tp_rank,
                                                    tp_world_size)


def _intersect_tensors(model: nn.Module,
                       available_tensors: dict[str, str]) -> set[str]:
    """
    Returns a set of files that parameters need to be loaded from.
    """
    result = set()
    for pname, _ in model.named_parameters():
        if pname in available_tensors:
            result.add(pname)
    return result


def _load_tensor_fully_or_partially(f, key: str,
                                    params: dict[str, torch.nn.Parameter],
                                    tp_rank: int, tp_world_size: int):
    """
    Loads a tensor from a safetensors file.

    :param f: The safetensors file.
    :param key: The key of the tensor to load.
    :param params: The parameters of the model.
    :param tp_rank: The rank of the tensor parallelism group.
    :param tp_world_size: The world size of the tensor parallelism group.
    """
    slc = f.get_slice(key)
    shape = slc.get_shape()
    param = params[key]
    if isinstance(param.data, DTensor):  # Requires partial load
        param_shape = param.data._local_tensor.shape
        diffs = [1 if (s != sts) else 0 for s, sts in zip(param_shape, shape)]
        if sum(diffs) == 0:  # No tensor parallelism
            param[:] = slc[:]
            return

        # Tensor parallelism (1D)
        if sum(diffs) > 1:
            raise ValueError('Only 1D parallelism is currently supported')
        tp_dim = next(d == 1 for d in diffs)

        # Get the total size and compute slice offset
        chunk_size = param.shape[tp_dim] // tp_world_size
        chunk_offset = tp_rank * chunk_size

        # Prepare slice
        # Use parameter shape to account for uneven distribution across ranks
        ndslice = [slice(None, None, None)] * len(shape)
        ndslice[tp_dim] = slice(chunk_offset,
                                chunk_offset + param.shape[tp_dim], 1)

        # Copy slice
        local_data = param.data._local_tensor
        local_data[:] = slc[tuple(ndslice)]
    else:
        # Full load
        param[:] = slc[:]
