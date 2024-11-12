import math

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from llama import load_checkpoint as lc


class LlamaDeviceMesh(DeviceMesh):
    """
    A device mesh subclass for Llama tensor and pipeline parallelism.
    """

    def __init__(self, tensor_parallel: int, pipeline_parallel: int = 1):
        """
        Create a device mesh for tensor and pipeline parallelism.

        :param tensor_parallel: The number of tensor parallel processes.
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

        :param name_or_path: The name or path of the pre-trained model
        :param device: The device to load the model on
        :param device_mesh: The device mesh for tensor and pipeline parallelism
        :param dtype: The data type for the model, defaults to torch.bfloat16
        :param delay_init: Whether to delay initialization until after sharding weights, defaults to True
        :param load_checkpoint: Whether to load from a checkpoint, defaults to False
        :param seed: The random seed for initialization, defaults to 0
        """
        super().__init__()
        self.device_mesh = device_mesh

        # Create the model and load from a checkpoint if needed
        init_device = torch.device("meta") if delay_init else device
        with init_device:
            config = LlamaConfig.from_pretrained(name_or_path)
            self.model = LlamaForCausalLM(config)
            self.model.to(dtype)
            self.model.eval()

        # Setup tensor parallel model sharding
        if device_mesh.tp_size() > 1:
            self._shard_model()

        # Setup pipeline parallelism
        if device_mesh.pp_size() > 1:
            self._pipeline_model()

        # Realize the model weights, if needed
        if delay_init:
            self.model.to_empty(device=device)

        # Ensure all ranks have the same seed for generation
        torch.manual_seed(seed)

        if load_checkpoint:
            lc.load_checkpoint(
                self.model,
                name_or_path,
                device_mesh.tp_rank(),
                device_mesh.tp_size(),
                device,
            )

    def _shard_model(self):
        """
        Setup tensor parallel model sharding.
        """
        # Shard each block in the transformer
        for layer in self.model.model.layers:
            block_plan = {
                "input_layernorm": SequenceParallel(),
                "self_attn": PrepareModuleInput(
                    input_kwarg_layouts={"hidden_states": Shard(1)},
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
                    input_layouts=Shard(1),
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

    def _pipeline_model(self):
        """
        Setup pipeline parallelism.
        """
        # Get the ranks and sizes for tensor and pipeline parallelism
        tp_rank = self.device_mesh.tp_rank()
        pp_rank = self.device_mesh.pp_rank()
        pp_size = self.device_mesh.pp_size()

        # Get the local blocks for this process
        blocks = self.model.model.layers
        num_local_blocks = math.ceil(len(blocks) / pp_size)
        start_block = pp_rank * num_local_blocks
        end_block = min(start_block + num_local_blocks, len(blocks))
        local_blocks = blocks[start_block:end_block]

        # Setup recv hook for the first block
        if pp_rank > 0:

            def recv_hook(module, hidden_states):
                tensor = hidden_states[0]
                if isinstance(tensor, DTensor):
                    tensor = tensor._local_tensor
                dist.batch_isend_irecv(
                    [
                        dist.P2POp(
                            dist.irecv,
                            tensor,
                            self.device_mesh.coord_to_rank(pp_rank - 1, tp_rank),
                        )
                    ]
                )[0].wait()

            local_blocks[0].register_forward_pre_hook(recv_hook)

        # Setup send hook for the last block
        if pp_rank < pp_size - 1:

            def send_hook(module, in_hidden_states, out_hidden_states):
                tensor = out_hidden_states[0]
                if isinstance(tensor, DTensor):
                    tensor = tensor._local_tensor
                dist.batch_isend_irecv(
                    [
                        dist.P2POp(
                            dist.isend,
                            tensor,
                            self.device_mesh.coord_to_rank(pp_rank + 1, tp_rank),
                        )
                    ]
                )[0].wait()

            local_blocks[-1].register_forward_hook(send_hook)

        # Brodcast final output to all processes in the pipeline parallel group
        def broadcast_hook(module, hidden_states):
            tensor = hidden_states[0]
            if isinstance(tensor, DTensor):
                tensor = tensor._local_tensor
            dist.broadcast(
                tensor,
                src=self.device_mesh.coord_to_rank(-1, tp_rank),
                group=self.device_mesh["pp"].get_group(),
            )

        self.model.model.norm.register_forward_pre_hook(broadcast_hook)

        # Replace blocks not in this process with an identity module
        class CustomIdentity(nn.Identity):
            def forward(self, hidden_states, **kwargs):
                output = (hidden_states,)
                if kwargs["output_attentions"]:
                    output += (None,)
                if kwargs["use_cache"]:
                    output += (kwargs["past_key_value"],)
                return output

        for i, block in enumerate(self.model.model.layers):
            if block not in local_blocks:
                self.model.model.layers[i] = CustomIdentity()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
