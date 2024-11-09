import math

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import (
    Replicate,
    Shard,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from transformers.models.llama import LlamaConfig, LlamaForCausalLM


class LlamaDeviceMesh(DeviceMesh):
    """
    A device mesh subclass for Llama tensor and pipeline parallelism.
    """

    def __init__(self, tensor_parallel: int, pipeline_parallel: int = 1):
        """
        Create a device mesh for tensor and pipeline parallelism.

        Args:
            tensor_parallel (int): The number of tensor parallel processes.
            pipeline_parallel (int, optional): The number of pipeline parallel processes. Defaults to 1.
        """
        assert pipeline_parallel == 1, "pipepline parallelism is not yet implemented"
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

        Returns:
            int: The rank of the current process in the tensor parallel group.
        """
        return self["tp"].get_local_rank()

    def tp_size(self):
        """
        Returns the size of the tensor parallel group.

        Returns:
            int: The size of the tensor parallel group.
        """
        return self["tp"].size()

    def pp_rank(self):
        """
        Returns the rank of the current process in the pipeline parallel group.

        Returns:
            int: The rank of the current process in the pipeline parallel group.
        """
        return self["pp"].get_local_rank()

    def pp_size(self):
        """
        Returns the size of the pipeline parallel group.

        Returns:
            int: The size of the pipeline parallel group.
        """
        return self["pp"].size()


class DistributedLlama(nn.Module):
    """
    A wrapper for the Hugging Face Llama model that distributes it across multiple devices using tensor and pipeline parallelism.
    """

    def __init__(
        self,
        name_or_path: str,
        device: torch.device,
        device_mesh: LlamaDeviceMesh,
        dtype: torch.dtype = torch.bfloat16,
        delay_init: bool = True,
        load_checkpoint: bool = False,
        seed: int = 0,
    ):
        """
        Create a distributed Llama model.

        Args:
            name_or_path (str): The name or path of the pre-trained model.
            device (torch.device): The device to load the model on.
            device_mesh (LlamaDeviceMesh): The device mesh for tensor and pipeline parallelism.
            dtype (torch.dtype, optional): The data type for the model. Defaults to torch.bfloat16.
            delay_init (bool, optional): Whether to delay initialization until after sharding weights. Defaults to True.
            load_checkpoint (bool, optional): Whether to load from a checkpoint. Defaults to False.
            seed (int, optional): The random seed for initialization. Defaults to 0.
        """
        super().__init__()
        self.device_mesh = device_mesh

        # Create the model and load from a checkpoint if needed
        init_device = torch.device("meta") if delay_init else device
        with init_device:
            if load_checkpoint:
                assert not delay_init, "delay_init must be False when loading checkpoint until sharded checkpoint loading is implemented"
                self.model = LlamaForCausalLM.from_pretrained(name_or_path)
            else:
                config = LlamaConfig.from_pretrained(name_or_path)
                self.model = LlamaForCausalLM(config)
                self.model.to(dtype)
                self.model.eval()

        # Setup tensor parallel model sharding
        self._shard_model()

        # Realize the model weights, if needed
        if delay_init:
            self.model.to_empty(device=device)

        # Ensure all ranks have the same seed for generation
        torch.manual_seed(seed)

    def _shard_model(self):
        """
        Setup tensor parallel model sharding.
        """
        # Shard each block in the transformer
        for layer in self.model.model.layers:
            block_plan = {
                "input_layernorm": SequenceParallel(),
                "self_attn": PrepareModuleInput(
                    desired_input_kwarg_layouts={"hidden_states": Replicate()},
                ),
                "self_attn.q_proj": ColwiseParallel(),
                "self_attn.k_proj": ColwiseParallel(),
                "self_attn.v_proj": ColwiseParallel(),
                "self_attn.o_proj": RowwiseParallel(
                    output_layouts=Shard(1), use_local_output=False
                ),
                "post_attention_layernorm": SequenceParallel(),
                "mlp": PrepareModuleInput(
                    desired_input_layouts=Replicate(),
                ),
                "mlp.gate_proj": ColwiseParallel(),
                "mlp.up_proj": ColwiseParallel(),
                "mlp.down_proj": RowwiseParallel(
                    output_layouts=Shard(1), use_local_output=False
                ),
            }
            parallelize_module(layer, self.device_mesh["tp"], block_plan)

            # Adjust the number of local heads
            layer.self_attn.num_heads = (
                layer.self_attn.num_heads // self.device_mesh.tp_size()
            )
            layer.self_attn.num_key_value_heads = (
                layer.self_attn.num_key_value_heads // self.device_mesh.tp_size()
            )

        # Shard the model embedding and output layers
        model_plan = {
            "model.embed_tokens": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
                use_local_output=False,
            ),
            "model.norm": SequenceParallel(),
            "lm_head": ColwiseParallel(output_layouts=Replicate()),
        }
        parallelize_module(self.model, self.device_mesh["tp"], model_plan)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
