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

import asyncio
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
    ControlMessageType,
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

    def __init__(
        self, tokenizer, queue, message_queue, skip_prompt=True, **decode_kwargs
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.queue = queue
        self.message_queue = message_queue
        self._prev_token = None

    def put(self, value):
        # Peek at the message queue to see if there are any outstanding messages
        if not self.message_queue.empty():
            self.message_queue.get()
            chat_synchronize_ranks(
                inputs, device, ControlInfo(message=ControlMessageType.CANCEL)
            )
            raise StopIteration
        else:
            chat_synchronize_ranks(inputs, device)

        if self.skip_prompt and self.next_tokens_are_prompt:
            # Skip both the prompt and the content header (in LLaMA 3.1, the
            # sequence separator is 128007 followed by '\n\n' which is 271)
            if value.shape[-1] == 1:  # Skip until start of answer
                if self._prev_token is None and value.item() < 128000:
                    self.next_tokens_are_prompt = False
                elif self._prev_token == 128007 and value.item() == 271:
                    self.next_tokens_are_prompt = False
                    return
                else:
                    self._prev_token = value.item()
                    return
            else:
                return

        v = self.tokenizer.batch_decode(value, skip_special_tokens=True)

        response = {"choices": [{"delta": {"role": "assistant", "content": str(v[0])}}]}
        self.queue.put(f"data: {json.dumps(response)}\n\n")

    def on_finalized_text(self, _: str, stream_end: bool = False):
        if stream_end:
            self._prev_token = None


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
    message_queue = queue.Queue()
    streamer = ChatServerTextStreamer(tokenizer, streamer_queue, message_queue)
    gen_queue.put((input_len, max_tokens, settings, streamer))

    # Return a streaming response
    if stream:

        async def content_stream(request: Request):
            try:
                while True:
                    # Check if the client has disconnected
                    if await request.is_disconnected():
                        print("Client has disconnected")
                        message_queue.put(None)
                        streamer_queue.get()  # Get final signal
                        break

                    res = streamer_queue.get()
                    if res is None:
                        break
                    yield res

            except asyncio.CancelledError:
                print("Chat stream was interrupted")
                message_queue.put(None)
                streamer_queue.get()  # Get final signal

        # Return a streaming response
        return StreamingResponse(
            content=content_stream(request), media_type="text/event-stream"
        )
    else:
        strip_str = 'data: {"choices": [{"delta": {"role": "assistant", "content": "'
        outputs = []
        # Collect all outputs
        while True:
            res = streamer_queue.get()
            if res is None:
                break
            # res is a string that looks like a json object e.g.
            # res = 'data: {"choices": [{"delta": {"role": "assistant", "content": "?"}}]}'
            # but we want to store just the content
            outputs.append(res.split(strip_str)[1][:-7])

        msg = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "".join(outputs),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "completion_tokens": len(outputs),
                "prompt_tokens": input_len,
                "total_tokens": (input_len + len(outputs)),
            },
        }
        # Return the collected outputs as a single response
        return msg


class EventLoopTextStreamer(TextStreamer):
    """
    Event loop streamer, which is called on every token generated in the generate
    call below. This streamer is used to synchronize ranks and handle control
    messages, in order to gracefully exit the chat loop.
    """

    def put(self, value):
        # Synchronize every token
        info: ControlInfo = chat_synchronize_ranks(inputs, device)
        if info is not None:
            if info.message == ControlMessageType.CANCEL:
                raise StopIteration(info)
            elif info.message == ControlMessageType.KEEPALIVE:
                # Skip keepalive messages
                return
            elif info.message == ControlMessageType.EXIT:
                raise StopIteration(info)


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
                chat_synchronize_ranks(
                    inputs, device, ControlInfo(message=ControlMessageType.EXIT)
                )
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
            chat_synchronize_ranks(
                inputs, device, ControlInfo(message=ControlMessageType.KEEPALIVE)
            )
        except StopIteration:  # Chat interrupted
            # Clear KV cache on interruption
            cache_manager.clear()

            # Send signal to end the stream
            streamer.queue.put(None)


def worker_loop():
    cache_manager = KVCacheManager(model)
    info: ControlInfo = chat_synchronize_ranks(inputs, device)
    while info.message != ControlMessageType.EXIT:
        if info.message != ControlMessageType.KEEPALIVE:
            kwargs = info.to_kwargs()

            try:
                outputs = model.generate(
                    input_ids=inputs[:, : info.input_len],
                    attention_mask=torch.ones((1, info.input_len), device=device),
                    streamer=EventLoopTextStreamer(tokenizer),
                    max_new_tokens=info.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    past_key_values=cache_manager.get_cache(
                        inputs, info.input_len, info.max_new_tokens
                    ),
                    **kwargs,
                )
                cache_manager.update(outputs)
            except StopIteration as ex:  # Chat interrupted
                info = ex.value
                if info is not None and info.message == ControlMessageType.EXIT:
                    break
                elif info is not None and info.message == ControlMessageType.CANCEL:
                    # Clear KV cache on interruption
                    cache_manager.clear()

        info = chat_synchronize_ranks(inputs, device)


def main():
    args = get_args(server=True)

    if not dist.is_initialized():
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
        model.model.forward = torch.compile(model.model.forward)

        if args.static_cache_size > 0:
            model.model.original_forward = model.model.forward
            model.model.static_cache_forward = torch.compile(
                model.model.forward, mode="reduce-overhead", dynamic=True
            )
            model.static_cache_size = args.static_cache_size

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
