import argparse

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from llama import DistributedLlama, LlamaDeviceMesh, load_checkpoint


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
    return parser.parse_args()


def main():
    args = get_args()

    device = torch.device("cuda:0")
    dist.init_process_group("nccl")
    device_mesh = LlamaDeviceMesh(tensor_parallel=dist.get_world_size())

    if args.model_size in ["8B", "70B"]:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = DistributedLlama(args.model_dir,
                                 device,
                                 device_mesh,
                                 delay_init=False,
                                 load_checkpoint=True)
    else:
        name = "meta-llama/Llama-3.1-405B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = DistributedLlama(args.model_dir, device, device_mesh, delay_init=True, load_checkpoint=False)
        print('Loading checkpoint...')
        with torch.no_grad():
            load_checkpoint.load_checkpoint(model.model, args.model_dir,
                                            device_mesh.tp_rank(),
                                            device_mesh.tp_size(), device)
        print('Done loading checkpoint')

    inputs = tokenizer("What is Apple?", return_tensors="pt").to(device)
    outputs = tokenizer.batch_decode(model.generate(**inputs,
                                                    max_new_tokens=100),
                                     skip_special_tokens=True)

    if dist.get_rank() == 0:
        print(outputs)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
