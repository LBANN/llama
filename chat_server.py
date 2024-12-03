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

import json
import queue
import threading
import time

import torch
import torch.distributed as dist
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer, TextStreamer

from llama import DistributedLlama, LlamaDeviceMesh
from llama.chat_utils import (
    ControlInfo,
    KVCacheManager,
    barrier,
    chat_synchronize_ranks,
    get_args,
)

# Restore affinity
Process().cpu_affinity(affinity)

# Create a FastAPI app
app = FastAPI()

# Global variables
inputs = model = tokenizer = gen_queue = None
device = torch.device("cuda:0")


class ChatServerTextStreamer(TextStreamer):
    """
    Text streamer that interacts with a streaming response for a chat server.
    """

    def __init__(self, tokenizer, queue, skip_prompt=True, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.queue = queue
        self._prev_token = None

    def put(self, value):
        if self.skip_prompt and self.next_tokens_are_prompt:
            # Skip both the prompt and the content header (in LLaMA 3.1, the
            # sequence separator is 128007 followed by '\n\n' which is 271)
            if value.shape[-1] == 1:  # Skip until start of answer
                if self._prev_token == 128007 and value.item() == 271:
                    self.next_tokens_are_prompt = False
                else:
                    self._prev_token = value.item()
            return

        v = self.tokenizer.batch_decode(value, skip_special_tokens=True)

        response = {"choices": [{"delta": {"role": "assistant", "content": str(v[0])}}]}
        self.queue.put(f"data: {json.dumps(response)}\n\n")


# Define a route for OpenAI API compatibility
@app.post("/chat/completions")
async def completions(request: Request):
    # Read the request body as JSON
    request_body = await request.json()

    # Handle the request
    messages = request_body.get("messages", [])
    max_tokens = request_body.get("max_tokens", 512)
    stream = request_body.get("stream", False)
    settings = {}
    if "temperature" in request_body:
        settings["temperature"] = request_body.get("temperature")

    actual_inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
    ).to(device)
    inputs[0, : actual_inputs.shape[-1]] = actual_inputs
    input_len = actual_inputs.shape[-1]

    streamer_queue = queue.Queue()
    streamer = ChatServerTextStreamer(tokenizer, streamer_queue)
    gen_queue.put((input_len, max_tokens, settings, streamer))

    def content_stream():
        while True:
            res = streamer_queue.get()
            if res is None:
                break
            yield res

    # Return a streaming response
    if stream:
        return StreamingResponse(
            content=content_stream(), media_type="text/event-stream"
        )
    else:
        raise NotImplementedError("Non-streaming completions are not supported")


@app.get("/models")
async def models():
    return {
        "data": [
            {
                "id": "LLLama",
                "name": "LLLama",
                "description": "Simple model",
            },
        ]
    }


def master_loop(inputs, device, gen_queue, interval_minutes=5):
    cache_manager = KVCacheManager(model)
    while True:
        try:
            task = gen_queue.get(timeout=interval_minutes * 60)
            # Check for shutdown signal
            if task is None:
                chat_synchronize_ranks(inputs, device, ControlInfo(exit=True))
                return
            input_len, max_tokens, settings, streamer = task

            # Synchronize the input tokens and lengths
            control_info = ControlInfo(
                input_len=input_len,
                max_new_tokens=max_tokens,
                temperature=settings.get("temperature", None),
            )
            chat_synchronize_ranks(inputs, device, control_info)
            kwargs = control_info.to_kwargs()

            # Generate text as a streaming response
            outputs = model.generate(
                input_ids=inputs[:, :input_len],
                attention_mask=torch.ones((1, input_len), device=device),
                streamer=streamer,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                past_key_values=cache_manager.get_cache(inputs, input_len, max_tokens),
                **kwargs,
            )

            # Update the cached tokens
            cache_manager.update(outputs)

            # Send signal to end the stream
            streamer.queue.put(None)
        except queue.Empty:
            # Send a keepalive signal
            chat_synchronize_ranks(inputs, device, ControlInfo(keepalive=True))


def worker_loop():
    cache_manager = KVCacheManager(model)
    info: ControlInfo = chat_synchronize_ranks(inputs, device)
    while not info.exit:
        if not info.keepalive:
            kwargs = info.to_kwargs()

            outputs = model.generate(
                input_ids=inputs[:, : info.input_len],
                attention_mask=torch.ones((1, info.input_len), device=device),
                streamer=None,
                max_new_tokens=info.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                past_key_values=cache_manager.get_cache(
                    inputs, info.input_len, info.max_new_tokens
                ),
                **kwargs,
            )
            cache_manager.update(outputs)
        info = chat_synchronize_ranks(inputs, device)


def main():
    args = get_args(server=True)

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

    global model
    global tokenizer
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

    global inputs
    inputs = torch.full((1, 131072), 128002, dtype=torch.long, device=device)

    if args.compile:
        model.model.original_forward = model.model.forward
        model.model.compiled_forward = torch.compile(
            model.model.forward, mode="reduce-overhead", dynamic=True
        )

    # Run the uvicorn server
    if dist.get_rank() == 0:
        # Start the keepalive thread
        global gen_queue
        gen_queue = queue.Queue()
        gen_thread = threading.Thread(
            target=master_loop, args=(inputs, device, gen_queue), daemon=True
        )
        gen_thread.start()

        # Detect the hostname and print it
        import socket

        print("Running server on", socket.gethostname())

        uvicorn.run(app, host=socket.gethostname(), port=args.port)

        print("Loop is over")

        # Send shutdown signal to main thread
        gen_queue.put(None)
    else:
        # Other ranks participate in the chat server by waiting
        worker_loop()


# Run the app
if __name__ == "__main__":
    main()
