import math

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
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
from transformers import AutoConfig
from transformers.models.llama import LlamaConfig, LlamaForCausalLM


class LlamaDeviceMesh(DeviceMesh):
    def __init__(self, tensor_parallel: int, pipeline_parallel: int = 1):
        assert pipeline_parallel == 1, "pipepline parallelism is not yet implemented"
        assert (
            tensor_parallel * pipeline_parallel == dist.get_world_size()
        ), "world size must be equal to the product of tensor and pipeline parallelism"
        mesh_shape = (pipeline_parallel, tensor_parallel)
        with torch.device("cpu"):
            mesh = torch.arange(math.prod(mesh_shape), dtype=torch.int).view(mesh_shape)
        super().__init__("cuda", mesh, mesh_dim_names=["pp", "tp"])

    def tp_rank(self):
        return self["tp"].get_local_rank()

    def tp_size(self):
        return self["tp"].size()

    def pp_rank(self):
        return self["pp"].get_local_rank()

    def pp_size(self):
        return self["pp"].size()


class DistributedLlama(nn.Module):
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
        super().__init__()
        self.device_mesh = device_mesh

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

        self._shard_model()

        if delay_init:
            self.model.to_empty(device=device)

        torch.manual_seed(seed)

    def _shard_model(self):
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

            layer.self_attn.num_heads = (
                layer.self_attn.num_heads // self.device_mesh["tp"].size()
            )
            layer.self_attn.num_key_value_heads = (
                layer.self_attn.num_key_value_heads // self.device_mesh["tp"].size()
            )

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
