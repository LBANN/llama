import argparse

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from llama import DistributedLlama, LlamaDeviceMesh
from llama.streaming import MasterRankTextStreamer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-size",
        type=str,
        default="405B",
        choices=["8B", "70B", "405B"],
        required=True,
    )
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

    if args.compile and device_mesh.tp_size() > 1:
        raise ValueError("torch.compile and tensor parallelism are mutually exclusive")

    if args.model_size in ["8B", "70B"]:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = DistributedLlama(
            args.model_dir,
            device,
            device_mesh,
            delay_init=False,
            load_checkpoint=not args.benchmark,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = DistributedLlama(
            args.model_dir,
            device,
            device_mesh,
            delay_init=True,
            load_checkpoint=not args.benchmark,
        )

    dist.all_reduce(torch.tensor(0, device=device), op=dist.ReduceOp.SUM)

    # Print how much memory is used by the GPU
    if dist.get_rank() == 0:
        print("Memory used:", torch.cuda.memory_allocated() / 1024**3, "GiB")

    inputs = tokenizer("What is DaCe?", return_tensors="pt").to(device)
    output_streamer = MasterRankTextStreamer(tokenizer, skip_special_tokens=True)

    if args.compile:
        model.model = torch.compile(model.model, mode="reduce-overhead")

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
