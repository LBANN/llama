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
import math

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh


class LlamaDeviceMesh(DeviceMesh):
    """
    A device mesh subclass for Llama tensor and pipeline parallelism.
    """

    def __init__(self, tensor_parallel: int = 1, pipeline_parallel: int = 1):
        """
        Create a device mesh for tensor and pipeline parallelism.

        :param tensor_parallel: The number of tensor parallel processes. Defaults to 1.
        :param pipeline_parallel: The number of pipeline parallel processes. Defaults to 1.
        """
        assert (
            tensor_parallel * pipeline_parallel == dist.get_world_size()
        ), "world size must be equal to the product of tensor and pipeline parallelism"
        mesh_shape = (pipeline_parallel, tensor_parallel)
        with torch.device("cpu"):
            mesh = torch.arange(math.prod(mesh_shape), dtype=torch.int).view(mesh_shape)
        super().__init__("cuda", mesh, mesh_dim_names=["pp", "tp"])

    def tp_rank(self):
        """
        Returns the rank of the current process in the tensor parallel group.

        :return: The rank of the current process in the tensor parallel group
        """
        return self["tp"].get_local_rank()

    def tp_size(self):
        """
        Returns the size of the tensor parallel group.

        :return: The size of the tensor parallel group
        """
        return self["tp"].size()

    def pp_rank(self):
        """
        Returns the rank of the current process in the pipeline parallel group.

        :return: The rank of the current process in the pipeline parallel group
        """
        return self["pp"].get_local_rank()

    def pp_size(self):
        """
        Returns the size of the pipeline parallel group.

        :return: The size of the pipeline parallel group
        """
        return self["pp"].size()

    def coord_to_rank(self, pp: int, tp: int):
        """
        Returns the rank of the process at the given coordinates in the device mesh.

        :param pp: The pipeline parallel coordinate
        :param tp: The tensor parallel coordinate
        :return: The rank of the process at the given coordinates
        """
        return self.mesh[pp, tp].item()


def parallelize_module(module: nn.Module, device_mesh: LlamaDeviceMesh, plan: dict):
    """
    Parallelize a module based on the given plan.

    :param module: The module to parallelize
    :param device_mesh: The device mesh for parallelism
    :param plan: The parallelism plan
    """
    for n, m in module.named_modules():
        if n in plan:
            plan[n].apply(m, device_mesh)


class SequenceParallel:
    """
    This class is used to parallelize a module in the sequence dimension.

    Inputs and outputs are replicated across all processes.
    """

    @staticmethod
    def apply(module: nn.Module, device_mesh: LlamaDeviceMesh):
        tp_rank = device_mesh.tp_rank()
        tp_size = device_mesh.tp_size()
        tp_group = device_mesh["tp"].get_group()

        # Setup input sharding hook
        def shard_hook(module, input):
            return input.chunk(tp_size, dim=1)[tp_rank]

        module.register_forward_pre_hook(shard_hook)

        # Setup gather hook for the output
        def gather_hook(module, input, output):
            outputs = [torch.empty_like(output) for _ in range(tp_size)]
            dist.all_gather(outputs, output, group=tp_group)
            return torch.cat(outputs, dim=1)

        module.register_forward_hook(gather_hook)


class EmbedParallel:
    """
    This class is used to parallelize an embedding module.

    Inputs and outputs are replicated across all processes.
    """

    @staticmethod
    def apply(module: nn.Module, device_mesh: LlamaDeviceMesh):
        if isinstance(module, nn.Embedding):
            # Shard the embeddings in a colwise fashion
            module.register_parameter(
                "weight",
                nn.Parameter(
                    module.weight.chunk(device_mesh.tp_size(), dim=-1)[
                        device_mesh.tp_rank()
                    ]
                ),
            )

            tp_size = device_mesh.tp_size()
            tp_group = device_mesh["tp"].get_group()

            # Setup all-gather hook for the output
            def all_gather_hook(module, input, output):
                outputs = [torch.empty_like(output) for _ in range(tp_size)]
                dist.all_gather(outputs, output, group=tp_group)
                return torch.cat(outputs, dim=-1)

            module.register_forward_hook(all_gather_hook)


class RowwiseParallel:
    """
    This class is used to parallelize a linear layer in the row dimension.

    Inputs are sharded along the last dimension and outputs are replicated across all processes.
    """

    @staticmethod
    def apply(module: nn.Module, device_mesh: LlamaDeviceMesh):
        if isinstance(module, nn.Linear):
            # Shard the weights in a rowwise fashion
            module.register_parameter(
                "weight",
                nn.Parameter(
                    module.weight.chunk(device_mesh.tp_size(), dim=-1)[
                        device_mesh.tp_rank()
                    ]
                ),
            )

            tp_group = device_mesh["tp"].get_group()

            # Setup all-reduce hook for the output
            def all_reduce_hook(module, input, output):
                dist.all_reduce(output, group=tp_group)

            module.register_forward_hook(all_reduce_hook)


class ColwiseParallel:
    """
    This class is used to parallelize a linear layer in the column dimension.

    Inputs are replicated across all processes and outputs are sharded along the last dimension.
    """

    @staticmethod
    def apply(module: nn.Module, device_mesh: LlamaDeviceMesh):
        if isinstance(module, nn.Linear):
            # Shard the weights in a colwise fashion
            module.register_parameter(
                "weight",
                nn.Parameter(
                    module.weight.chunk(device_mesh.tp_size(), dim=0)[
                        device_mesh.tp_rank()
                    ]
                ),
            )
