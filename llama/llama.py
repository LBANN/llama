import math

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from llama import load_checkpoint as lc
from llama.parallel import (
    ColwiseParallel,
    EmbedParallel,
    LlamaDeviceMesh,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)


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
            config = LlamaConfig.from_pretrained(
                name_or_path, torch_dtype=dtype, attn_implementation="sdpa"
            )
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

            with device:
                self.model.model.rotary_emb = LlamaRotaryEmbedding(config=config)

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
                "self_attn.q_proj": ColwiseParallel(),
                "self_attn.k_proj": ColwiseParallel(),
                "self_attn.v_proj": ColwiseParallel(),
                "self_attn.o_proj": RowwiseParallel(),
                "mlp.gate_proj": ColwiseParallel(),
                "mlp.up_proj": ColwiseParallel(),
                "mlp.down_proj": RowwiseParallel(),
            }
            parallelize_module(layer, self.device_mesh, block_plan)

            # Adjust the number of local heads
            layer.self_attn.num_heads = (
                layer.self_attn.num_heads // self.device_mesh.tp_size()
            )

            layer.self_attn.num_key_value_heads = (
                layer.self_attn.num_key_value_heads // self.device_mesh.tp_size()
            )

        # Shard the model embedding and output layers
        model_plan = {
            "model.embed_tokens": EmbedParallel(),
        }
        parallelize_module(self.model, self.device_mesh, model_plan)

        self.model.config.num_key_value_heads = (
            self.model.config.num_key_value_heads // self.device_mesh.tp_size()
        )

    def _pipeline_model(self):
        """
        Setup pipeline parallelism.
        """
        # Get the ranks and sizes for tensor and pipeline parallelism
        tp_rank = self.device_mesh.tp_rank()
        pp_rank = self.device_mesh.pp_rank()
        pp_size = self.device_mesh.pp_size()
        pp_group = self.device_mesh["pp"].get_group()

        # Get the local blocks for this process
        blocks = self.model.model.layers
        num_local_blocks = math.ceil(len(blocks) / pp_size)
        start_block = pp_rank * num_local_blocks
        end_block = min(start_block + num_local_blocks, len(blocks))
        local_blocks = blocks[start_block:end_block]

        # Setup recv hook for the first block
        if pp_rank > 0:
            src_rank = self.device_mesh.coord_to_rank(pp_rank - 1, tp_rank)

            def recv_hook(module, hidden_states):
                tensor = hidden_states[0]
                if isinstance(tensor, DTensor):
                    tensor = tensor._local_tensor
                dist.batch_isend_irecv([dist.P2POp(dist.irecv, tensor, src_rank)])[
                    0
                ].wait()

            local_blocks[0].register_forward_pre_hook(recv_hook)

        # Setup send hook for the last block
        if pp_rank < pp_size - 1:
            dst_rank = self.device_mesh.coord_to_rank(pp_rank + 1, tp_rank)

            def send_hook(module, in_hidden_states, out_hidden_states):
                tensor = out_hidden_states[0]
                if isinstance(tensor, DTensor):
                    tensor = tensor._local_tensor
                dist.batch_isend_irecv([dist.P2POp(dist.isend, tensor, dst_rank)])[
                    0
                ].wait()

            local_blocks[-1].register_forward_hook(send_hook)

        # Brodcast final output to all processes in the pipeline parallel group
        src_rank_for_bcast = self.device_mesh.coord_to_rank(-1, tp_rank)

        def broadcast_hook(module, hidden_states):
            tensor = hidden_states[0]
            if isinstance(tensor, DTensor):
                tensor = tensor._local_tensor
            dist.broadcast(
                tensor,
                src=src_rank_for_bcast,
                group=pp_group,
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
