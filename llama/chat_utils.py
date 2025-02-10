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
import argparse
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum

import torch
import torch.distributed as dist
from transformers import PreTrainedTokenizer
from transformers.cache_utils import DynamicCache, StaticCache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from llama import DistributedLlama
from llama.streaming import MasterRankTextStreamer


def get_args(server: bool = False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument(
        "--pp",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
    )
    parser.add_argument("--static-cache-size", action="store", type=int, default=0)
    parser.add_argument(
        "--max-tokens-per-response",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--io-threads",
        type=int,
        default=0,
    )
    if server:
        parser.add_argument("--port", type=int, default=8123)
    return parser.parse_args()


class ControlMessageType(Enum):
    NORMAL = 0
    EXIT = 1
    KEEPALIVE = 2
    CANCEL = 3


@dataclass
class ControlInfo:
    message: ControlMessageType = ControlMessageType.NORMAL
    input_len: int = 0
    max_new_tokens: int = 0
    temperature: float = None

    def to_kwargs(self):
        result = {}
        if self.temperature is not None:
            result["temperature"] = self.temperature

        return result


class PipelineDynamicCache(DynamicCache):
    """
    A custom dynamic cache that keeps track of the sequence length from all layers.
    The default implementation only keeps track of the sequence length for the first layer.
    """

    def get_seq_length(self, layer_idx: int = None) -> int:
        if layer_idx is None:
            if len(self.key_cache) == 0:
                return 0
            max_seq_length = max(
                [
                    layer_cache.shape[-2]
                    for layer_cache in self.key_cache
                    if len(layer_cache) > 0
                ]
            )
            return max_seq_length

        return super().get_seq_length(layer_idx)


class PipelineStaticCache(StaticCache):
    """
    A custom static cache that only caches the key and value states for layers that have not been
    pipelined out.
    """

    def __init__(
        self,
        model: DistributedLlama,
        max_batch_size: int = 1,
        dtype=torch.bfloat16,
        max_cache_len=2048,
    ):
        count = 0
        self.layer_map = [0] * len(model.model.model.layers)
        for i, layer in enumerate(model.model.model.layers):
            if isinstance(layer, LlamaDecoderLayer):
                self.layer_map[i] = count
                count += 1

        config = deepcopy(model.model.config)
        config.num_layers = count
        config.max_position_embeddings = max_cache_len
        self.max_batch_size = max_batch_size

        super().__init__(
            config,
            max_batch_size=max_batch_size,
            dtype=dtype,
            device=model.model.device,
        )

    def remap_layer(self, layer_idx: int):
        return self.layer_map[layer_idx]

    def get_seq_length(self, layer_idx: int = 0):
        layer_idx = self.remap_layer(layer_idx)
        return super().get_seq_length(layer_idx)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict = None,
    ):
        layer_idx = self.remap_layer(layer_idx)
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    def to_dynamic_cache(self, model: DistributedLlama):
        cache = PipelineDynamicCache()
        cache.key_cache = [[] for _ in range(len(self.layer_map))]
        cache.value_cache = [[] for _ in range(len(self.layer_map))]
        seq_len = self.get_seq_length()
        for i, layer in enumerate(model.model.model.layers):
            if isinstance(layer, LlamaDecoderLayer):
                cache.key_cache[i] = self.key_cache[self.remap_layer(i)][:, :, :seq_len]
                cache.value_cache[i] = self.value_cache[self.remap_layer(i)][
                    :, :, :seq_len
                ]

        return cache


class KVCacheManager:
    """
    Manages the key-value cache for the model, keeping track of the previous tokens.
    Dynamically switches between static and dynamic caches based on the input length.
    """

    def __init__(self, model: DistributedLlama):
        self.model = model
        self.use_static_cache = hasattr(self.model.model, "static_cache_forward")

        if self.use_static_cache:
            self.static_cache = PipelineStaticCache(
                self.model,
                max_cache_len=model.static_cache_size,
                dtype=model.model.dtype,
            )

        self.clear()

    def get_cache(self, inputs, input_len, max_new_tokens):
        if (
            self.cached_tokens is not None
            and self.cached_tokens.shape[0] != inputs.shape[0]
        ):
            print("Cache miss (batch size)")
            self.clear(batch_size=inputs.shape[0])
        else:
            # Check if the cache can be reused
            if self.cached_tokens is not None:
                cached_len = self.cached_tokens.shape[1]
                if not (
                    cached_len < input_len
                    and torch.equal(self.cached_tokens, inputs[:, :cached_len])
                ):
                    print("Cache miss")
                    self.clear(batch_size=inputs.shape[0])
                else:
                    print("Cache hit")

        # Switch to dynamic cache if the static cache is too small
        if isinstance(
            self.kv_cache, PipelineStaticCache
        ) and self.kv_cache.max_cache_len < (input_len + max_new_tokens):
            print("Switching to dynamic cache")
            self.kv_cache = self.kv_cache.to_dynamic_cache(self.model)

        # Switch to compiled forward if available
        if self.use_static_cache:
            if isinstance(self.kv_cache, PipelineStaticCache):
                print("Using static cache forward")
                self.model.model.forward = self.model.model.static_cache_forward
            else:
                print("Using original forward")
                self.model.model.forward = self.model.model.original_forward

        return self.kv_cache

    def update(self, outputs):
        self.cached_tokens = outputs

    def clear(self, batch_size: int = 1):
        self.cached_tokens = None

        if self.use_static_cache:
            # Perform a full reset of the static cache if the batch size changed
            if (
                self.static_cache is not None
                and batch_size != self.static_cache.max_batch_size
            ):
                self.static_cache = PipelineStaticCache(
                    self.model,
                    max_batch_size=batch_size,
                    max_cache_len=self.model.static_cache_size,
                    dtype=self.model.model.dtype,
                )
            else:
                self.static_cache.reset()
            self.kv_cache = self.static_cache
        else:
            self.kv_cache = PipelineDynamicCache()


def barrier(device):
    """
    A barrier that synchronizes all ranks and is compatible with DTensor.
    """
    # This is a "barrier" that is supported with a device mesh
    dist.all_reduce(torch.tensor(0, device=device), op=dist.ReduceOp.SUM)


def chat_synchronize_ranks(inputs, device, info=None):
    """
    Waits for all ranks to receive the input length of the next message.
    """
    barrier(device)
    info_list = [info]
    dist.broadcast_object_list(info_list, 0)
    dist.broadcast(inputs, 0)

    return info_list[0]


def chat_loop(
    model: DistributedLlama,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    output_streamer: MasterRankTextStreamer,
    args: argparse.Namespace,
):
    """
    Chat loop that synchronizes ranks to receive the same input and output.

    The user can type a message, which is then sent to the assistant model.
    The assistant model generates a response, which is then printed by the master rank.

    :param model: The assistant model.
    :param tokenizer: The tokenizer used for the model.
    :param device: The device used for the model.
    :param output_streamer: The streamer used to print the output.
    :param args: The command-line arguments
    """
    # Keep memory buffers for messages and inputs
    messages = []
    inputs = torch.full((1, 131072), 128002, dtype=torch.long, device=device)

    if args.debug and dist.get_rank() == 0:
        print("Warming up...")
    model.generate(
        input_ids=inputs[:, 0:128],
        attention_mask=torch.ones((1, 128), device=device),
        streamer=None,
        max_new_tokens=1,
        pad_token_id=tokenizer.eos_token_id,
        past_key_values=PipelineStaticCache(model),
    )

    if dist.get_rank() == 0:
        print(
            "Type a message to start the chat.",
            "Press ctrl-D or type 'exit' to end the conversation.",
        )

    try:
        # Loop forever
        while True:
            # Ask for inputs only on the first rank
            info = None
            if dist.get_rank() == 0:
                if args.benchmark:
                    message = "benchmark"
                else:
                    message = input("> ")
                if message.strip() == "exit":
                    raise EOFError

                messages.append({"role": "user", "content": message})
                actual_inputs = tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                ).to(device)
                inputs[0, : actual_inputs.shape[-1]] = actual_inputs
                info = ControlInfo(input_len=actual_inputs.shape[-1])

            # Synchronize the input tokens and lengths
            info = chat_synchronize_ranks(inputs, device, info)
            if info.message == ControlMessageType.EXIT:
                break

            # The crux of the chat loop: generate the response
            model.generate(
                input_ids=inputs[:, : info.input_len],
                attention_mask=torch.ones((1, info.input_len), device=device),
                streamer=output_streamer,
                max_new_tokens=10 if args.benchmark else args.max_tokens_per_response,
                pad_token_id=tokenizer.eos_token_id,
                past_key_values=PipelineStaticCache(model),
            )

            # Debug print performance
            if dist.get_rank() == 0 and (args.debug or args.benchmark):
                s_tok = torch.tensor(output_streamer.time_per_token)
                print("\nMedian tokens/s:", 1 / s_tok.median().item())
                output_streamer.time_per_token = []
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                total_gpu_memory = gpu_memory * dist.get_world_size()
                print(
                    f"Memory used: {gpu_memory:.2f} GiB per GPU [Total memory ~= {total_gpu_memory:.2f} GiB]"
                )
            # Benchmark only runs one iteration
            if args.benchmark:
                break

            # Keep track of the conversation
            messages.append(
                {"role": "assistant", "content": output_streamer.last_message}
            )
            output_streamer.clear_last_message()
    except (EOFError, KeyboardInterrupt):
        # Broadcast zeros to finalize the rest of the ranks
        if dist.get_rank() == 0:
            print("[Ending chat]")
            chat_synchronize_ranks(
                inputs, device, ControlInfo(message=ControlMessageType.EXIT)
            )
