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
from llama.chat_utils import barrier, chat_synchronize_ranks, get_args

# Create a FastAPI app
app = FastAPI()

# Global variables
input_len = inputs = model = tokenizer = lock = None
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
            if (
                value.shape[-1] == 1
            ):  # Skip until start of answer
                if self._prev_token == 128007 and value.item() == 271:
                    self.next_tokens_are_prompt = False
                else:
                    self._prev_token = value.item()
            return

        v = self.tokenizer.batch_decode(value, skip_special_tokens=True)

        response = {"choices": [{"delta": {"role": "assistant", "content": str(v[0])}}]}
        self.queue.put(f"data: {json.dumps(response)}\n\n")


def generate_text(streamer, max_tokens):
    with lock:
        # Synchronize the input tokens and lengths
        chat_synchronize_ranks(inputs, input_len, device)

        # Generate text as a streaming response
        model.generate(
            input_ids=inputs[:, : input_len[0]],
            attention_mask=torch.ones((1, input_len[0]), device=device),
            streamer=streamer,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Send signal to end the stream
        streamer.queue.put(None)


# Define a route for OpenAI API compatibility
@app.post("/chat/completions")
async def completions(request: Request):
    # Read the request body as JSON
    request_body = await request.json()

    # Handle the request
    messages = request_body.get("messages", [])
    max_tokens = request_body.get("max_tokens", 512)
    stream = request_body.get("stream", False)

    actual_inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
    ).to(device)
    inputs[0, : actual_inputs.shape[-1]] = actual_inputs
    input_len[0] = actual_inputs.shape[-1]
    input_len[1] = max_tokens

    streamer_queue = queue.Queue()
    streamer = ChatServerTextStreamer(tokenizer, streamer_queue)

    # Generate text
    threading.Thread(
        target=generate_text, args=(streamer, max_tokens), daemon=True
    ).start()

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


def event_loop():
    chat_synchronize_ranks(inputs, input_len, device)
    while input_len[0] >= 0:
        if input_len[0] > 0:
            model.generate(
                input_ids=inputs[:, : input_len[0]],
                attention_mask=torch.ones((1, input_len[0]), device=device),
                streamer=None,
                max_new_tokens=input_len[1],
                pad_token_id=tokenizer.eos_token_id,
            )
        chat_synchronize_ranks(inputs, input_len, device)


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


def keepalive_signal(inputs, input_len, device, lock, interval_minutes=5):
    """Send a keepalive signal from rank 0 to other ranks every `interval_minutes`."""
    inputs = torch.empty_like(inputs)
    input_len = torch.empty_like(input_len)
    input_len[0] = 0
    while True:
        time.sleep(interval_minutes * 60)
        with lock:
            chat_synchronize_ranks(inputs, input_len, device)


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

    # Warm up the model
    if args.debug and dist.get_rank() == 0:
        print("Warming up...")
    model.generate(
        input_ids=torch.full((1, 128), 128002, dtype=torch.long, device=device),
        attention_mask=torch.ones((1, 128), device=device),
        streamer=None,
        max_new_tokens=1,
        pad_token_id=tokenizer.eos_token_id,
    )

    global inputs, input_len
    inputs = torch.full((1, 131072), 128002, dtype=torch.long, device=device)
    input_len = torch.zeros((2,), dtype=torch.long, device=device)

    # Run the uvicorn server
    if dist.get_rank() == 0:
        # Start the keepalive thread
        global lock
        lock = threading.Lock()
        keepalive_thread = threading.Thread(
            target=keepalive_signal, args=(inputs, input_len, device, lock)
        )
        keepalive_thread.daemon = True
        keepalive_thread.start()

        # Detect the hostname and print it
        import socket

        print("Running server on", socket.gethostname())

        uvicorn.run(app, host="0.0.0.0", port=args.port)

        print("Loop is over")

        # Tear down the process group
        input_len[0] = -1
        chat_synchronize_ranks(inputs, input_len, device)
    else:
        # Other ranks participate in the chat server by waiting
        event_loop()


# Run the app
if __name__ == "__main__":
    main()
