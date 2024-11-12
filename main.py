import argparse

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, TextStreamer
import time

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


class MasterRankTextStreamer(TextStreamer):
    """
    Text streamer that only prints text from the master rank.
    For more information, see :class:`~transformers.TextStreamer`.
    """

    def __init__(self, tokenizer, skip_prompt=False, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.time_per_token = []
        self.last_time = time.time()

    def put(self, value):
        if dist.get_rank() != 0:
            return
        time_since_last_token = time.time() - self.last_time
        v = self.tokenizer.batch_decode(value, skip_special_tokens=True)
        print(v[0], end='', flush=True)
        self.time_per_token.append(time_since_last_token)
        self.last_time = time.time()

    # def on_finalized_text(self, text: str, stream_end: bool = False):
    #     if dist.get_rank() == 0:
    #         super().on_finalized_text(text, stream_end)


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
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = DistributedLlama(args.model_dir, device, device_mesh, delay_init=True, load_checkpoint=False)
        print('Loading checkpoint...')
        with torch.no_grad():
            load_checkpoint.load_checkpoint(model.model, args.model_dir,
                                            device_mesh.tp_rank(),
                                            device_mesh.tp_size(), device)
        print('Done loading checkpoint')

    inputs = tokenizer("What is Apple?", return_tensors="pt").to(device)
    output_streamer = MasterRankTextStreamer(tokenizer,
                                             skip_special_tokens=True)
    outputs = model.generate(**inputs,
                             streamer=output_streamer,
                             max_new_tokens=10,
                             pad_token_id=tokenizer.eos_token_id)

    if dist.get_rank() == 0:
        print('\n\nTime per token:', output_streamer.time_per_token)
    # if dist.get_rank() == 0:
    #     print(outputs)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
