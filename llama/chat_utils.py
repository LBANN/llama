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
from dataclasses import dataclass
from enum import Enum

import torch
import torch.distributed as dist
from transformers import PreTrainedTokenizer
from transformers.cache_utils import DynamicCache

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


class CustomDynamicCache(DynamicCache):
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


class KVCacheManager:
    """
    Manages the key-value cache for the model, keeping track of the previous tokens.
    """

    def __init__(self):
        self.cached_tokens = None
        self.kv_cache = CustomDynamicCache()

    def get_cache(self, inputs, input_len):
        if self.cached_tokens is not None:
            # Check if the cache can be reused
            cached_len = self.cached_tokens.shape[1]
            if cached_len < input_len and torch.equal(
                self.cached_tokens, inputs[:, :cached_len]
            ):
                return self.kv_cache
            else:
                self.cached_tokens = None
                self.kv_cache = CustomDynamicCache()
        return self.kv_cache

    def update(self, outputs):
        self.cached_tokens = outputs


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
