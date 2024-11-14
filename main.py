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

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from llama import DistributedLlama, LlamaDeviceMesh
from llama.streaming import MasterRankTextStreamer


def get_args():
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
        "--compile",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def main():
    args = get_args()

    device = torch.device("cuda:0")
    dist.init_process_group("nccl")
    device_mesh = LlamaDeviceMesh(
        tensor_parallel=dist.get_world_size() // args.pp, pipeline_parallel=args.pp
    )
    print(
        f"Device mesh: rank={dist.get_rank()}, TP={device_mesh.tp_rank()}/{device_mesh.tp_size()}, PP={device_mesh.pp_rank()}/{device_mesh.pp_size()}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = DistributedLlama(
        args.model_dir,
        device,
        device_mesh,
        delay_init=True,
        load_checkpoint=not args.benchmark,
    )

    # This is a "barrier" that is supported with a device mesh
    dist.all_reduce(torch.tensor(0, device=device), op=dist.ReduceOp.SUM)

    # Print how much memory is used by the GPU
    if dist.get_rank() == 0:
        print("Memory used:", torch.cuda.memory_allocated() / 1024**3, "GiB")

    inputs = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "What is LBANN?"},
        ],
        return_tensors="pt",
        return_dict=True,
    ).to(device)
    output_streamer = MasterRankTextStreamer(tokenizer, skip_special_tokens=True)

    if args.compile:
        model.model.forward = torch.compile(model.model.forward) #, mode="reduce-overhead")

    outputs = model.generate(
        **inputs,
        streamer=output_streamer,
        max_new_tokens=10 if args.benchmark else 100,
        pad_token_id=tokenizer.eos_token_id,
    )

    if dist.get_rank() == 0:
        s_tok = torch.tensor(output_streamer.time_per_token[1:])
        print("\nMedian tokens/s:", 1 / s_tok.median().item())
        # print(outputs)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
