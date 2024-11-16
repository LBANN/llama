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
from psutil import Process

# Save affinity
affinity = Process().cpu_affinity()

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from llama import DistributedLlama, LlamaDeviceMesh
from llama.chat_utils import *
from llama.streaming import MasterRankTextStreamer

# Restore affinity
Process().cpu_affinity(affinity)


def main():
    args = get_args()

    device = torch.device("cuda:0")
    dist.init_process_group("nccl")
    device_mesh = LlamaDeviceMesh(
        tensor_parallel=dist.get_world_size() // args.pp, pipeline_parallel=args.pp
    )
    if args.debug:
        print(
            f"Device mesh: rank={dist.get_rank()},",
            f"TP={device_mesh.tp_rank()}/{device_mesh.tp_size()},",
            f"PP={device_mesh.pp_rank()}/{device_mesh.pp_size()}",
        )

    # Choose the number of I/O threads automatically
    io_threads = args.io_threads if args.io_threads > 0 else device_mesh.tp_size()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = DistributedLlama(
        args.model_dir,
        device,
        device_mesh,
        delay_init=True,
        load_checkpoint=not args.benchmark,
        io_threads=io_threads,
    )
    barrier(device)

    # Print how much memory is used by the GPU
    if dist.get_rank() == 0 and (args.debug or args.benchmark):
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        total_gpu_memory = gpu_memory * dist.get_world_size()
        print(
            f"Memory used: {gpu_memory:.2f} GiB per GPU [Total memory ~= {total_gpu_memory:.2f} GiB]"
        )

    if args.compile:
        model.model.forward = torch.compile(
            model.model.forward
        )  # , mode="reduce-overhead")

    output_streamer = MasterRankTextStreamer(
        tokenizer, skip_special_tokens=True, skip_prompt=not args.benchmark
    )
    chat_loop(model, tokenizer, device, output_streamer, args)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
