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
from dataclasses import dataclass

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
model = tokenizer = None
device = torch.device("cuda:0")
max_batch_size = None
streaming_request_queue: queue.Queue = None
nonstreaming_request_queue: queue.Queue = None


@dataclass
class ChatRequest:
    """
    Object that stores a chat request.
    """
    inputs: list[int]
    max_tokens: int
    settings: dict[str, float]
    streamer_queue: queue.Queue
    message_queue: queue.Queue


class ChatServerTextStreamer(TextStreamer):
    """
    Text streamer that interacts with a streaming response for a chat server.
    """

    def __init__(
        self,
        tokenizer,
        queues: list[queue.Queue],
        message_queues: list[queue.Queue],
        skip_prompt: bool = True,
        **decode_kwargs,
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.queues = queues
        self.message_queues = message_queues
        self._prev_token = None

    def put(self, value: torch.Tensor):
        # Peek at the message queue to see if there are any outstanding messages
        if len(self.message_queues) == 1 and not self.message_queues[0].empty():
            # Cancelling prompts is only supported for batch size of 1
            self.message_queues[0].get()
            chat_synchronize_ranks(
                device, ControlInfo(message=ControlMessageType.CANCEL)
            )
            raise StopIteration
        else:
            chat_synchronize_ranks(device)

        if self.skip_prompt and self.next_tokens_are_prompt:
            # Skip both the prompt and the content header (in LLaMA 3.1, the
            # sequence separator is 128007 followed by '\n\n' which is 271)
            if value.ndim == 1:  # Skip until start of answer
                if self._prev_token is None and torch.all(value < 128000):
                    self.next_tokens_are_prompt = False
                elif self._prev_token == 128007 and torch.all(value == 271):
                    self.next_tokens_are_prompt = False
                    return
                else:
                    # Since all sequences are "right-justified", the first works
                    self._prev_token = value[0]
                    return
            else:
                return

        v = self.tokenizer.batch_decode(value, skip_special_tokens=True)

        for i, q in enumerate(self.queues):
            response = {
                "choices": [{"delta": {"role": "assistant", "content": str(v[i])}}]
            }
            q.put(f"data: {json.dumps(response)}\n\n")

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

    actual_inputs: torch.Tensor = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
    )
    inputs = actual_inputs.flatten().tolist()
    input_len = len(inputs)

    streamer_queue = queue.Queue()
    message_queue = queue.Queue()
    chat_request = ChatRequest(
        inputs, max_tokens, settings, streamer_queue, message_queue
    )

    # Return a streaming response
    if stream:
        streaming_request_queue.put(chat_request)
        await asyncio.sleep(0.1)  # Yield execution to allow batch size to accumulate

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
        nonstreaming_request_queue.put(chat_request)
        await asyncio.sleep(0.1)  # Yield execution to allow batch size to accumulate

        strip_str = 'data: {"choices": [{"delta": {"role": "assistant", "content": "'
        outputs = []
        # Collect all outputs
        output_tokens = 0
        while True:
            res = streamer_queue.get()
            if res is None:
                break
            # res is a string that looks like a json object e.g.
            # res = 'data: {"choices": [{"delta": {"role": "assistant", "content": "?"}}]}'
            # but we want to store just the content
            outputs.append(res.split(strip_str)[1][:-7])
            output_tokens += 1

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
                "completion_tokens": output_tokens,
                "prompt_tokens": input_len,
                "total_tokens": (input_len + output_tokens),
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
        info: ControlInfo = chat_synchronize_ranks(device)
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


def aggregate_tasks(
    request_queue: queue.Queue, max_batch_size: int, interval_minutes: int
):
    """
    Aggregate tasks from the request queue and process them in a batch.
    """
    requests: list[ChatRequest] = []
    # Process up to `max_batch_size` requests
    print("Queue size:", request_queue.qsize())
    while request_queue.qsize() > 0:
        requests.append(request_queue.get(timeout=interval_minutes * 60))

        # Check for shutdown signal
        if requests[-1] is None:
            return None

        if len(requests) >= max_batch_size:
            break

    # Aggregate the inputs and settings
    qlen = len(requests)
    input_len = max([len(x.inputs) for x in requests])
    actual_inputs = torch.full(
        (qlen, input_len), tokenizer.eos_token_id, device=device, dtype=torch.long
    )
    for i, request in enumerate(requests):
        actual_inputs[i, -len(request.inputs) :] = torch.tensor(request.inputs)

    max_tokens = max([x.max_tokens for x in requests])
    settings = {}
    for request in requests:
        settings.update(request.settings)

    message_queues = [x.message_queue for x in requests]
    streamer_queues = [x.streamer_queue for x in requests]
    input_lengths = [len(x.inputs) for x in requests]

    return (
        actual_inputs,
        max_tokens,
        settings,
        message_queues,
        streamer_queues,
        input_lengths,
    )


def master_loop(
    inputs,
    device,
    streaming_request_queue,
    nonstreaming_request_queue,
    batch_delay,
    interval_minutes=5,
):
    cache_manager = KVCacheManager(model)
    while True:
        try:
            if batch_delay > 0 and max_batch_size > 1:
                time.sleep(batch_delay / 1000)

            # Aggregate tasks from the request queues
            if not streaming_request_queue.empty():
                task = aggregate_tasks(
                    streaming_request_queue, max_batch_size, interval_minutes
                )
            elif not nonstreaming_request_queue.empty():
                task = aggregate_tasks(
                    nonstreaming_request_queue, max_batch_size, interval_minutes
                )
            else:
                continue

            # Check for shutdown signal
            if task is None:
                chat_synchronize_ranks(
                    device, ControlInfo(message=ControlMessageType.EXIT)
                )
                return

            (
                inputs,
                max_tokens,
                settings,
                message_queues,
                streamer_queues,
                input_lengths,
            ) = task
            input_len = inputs.shape[1]

            # Synchronize the input tokens and lengths
            control_info = ControlInfo(
                batch_size=inputs.shape[0],
                input_len=inputs.shape[1],
                max_new_tokens=max_tokens,
                temperature=settings.get("temperature", None),
            )
            chat_synchronize_ranks(device, control_info)
            kwargs = control_info.to_kwargs()
            dist.broadcast(inputs, 0)

            # Prepare attention mask based on input lengths
            attention_mask = torch.ones_like(inputs)
            for i, length in enumerate(input_lengths):
                if length < input_len:
                    attention_mask[i, 0 : input_len - length] = 0
            dist.broadcast(attention_mask, 0)

            streamer = ChatServerTextStreamer(
                tokenizer, streamer_queues, message_queues
            )

            if inputs.shape[0] > 1:
                print("Batched request. Batch size:", inputs.shape[0])

            # Generate text as a streaming response
            outputs = model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                streamer=streamer,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                past_key_values=cache_manager.get_cache(inputs, input_len, max_tokens),
                **kwargs,
            )

            # Update the cached tokens
            cache_manager.update(outputs)

            # Send signal to end the stream
            for q in streamer.queues:
                q.put(None)
        except queue.Empty:
            # Send a keepalive signal
            chat_synchronize_ranks(
                device, ControlInfo(message=ControlMessageType.KEEPALIVE)
            )
        except StopIteration:  # Chat interrupted
            # Clear KV cache on interruption
            cache_manager.clear()

            # Send signal to end the stream
            for q in streamer.queues:
                q.put(None)


def worker_loop():
    cache_manager = KVCacheManager(model)
    info: ControlInfo = chat_synchronize_ranks(device)
    while info.message != ControlMessageType.EXIT:
        if info.message != ControlMessageType.KEEPALIVE:
            kwargs = info.to_kwargs()

            # Synchronize the input tokens and lengths
            inputs = torch.empty(
                (info.batch_size, info.input_len), device=device, dtype=torch.long
            )
            attention_mask = torch.empty_like(inputs)
            dist.broadcast(inputs, 0)
            dist.broadcast(attention_mask, 0)

            try:
                outputs = model.generate(
                    input_ids=inputs,
                    attention_mask=attention_mask,
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

        info = chat_synchronize_ranks(device)


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

    global max_batch_size
    max_batch_size = args.max_batch_size

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
        # Create request queues for users
        global streaming_request_queue
        global nonstreaming_request_queue
        streaming_request_queue = queue.Queue()
        nonstreaming_request_queue = queue.Queue()

        # Start the keepalive thread
        gen_thread = threading.Thread(
            target=master_loop,
            args=(
                inputs,
                device,
                streaming_request_queue,
                nonstreaming_request_queue,
                args.batch_delay,
            ),
            daemon=True,
        )
        gen_thread.start()

        # Detect the hostname and print it
        import socket

        print("Running server on", socket.gethostname())

        uvicorn.run(app, host=socket.gethostname(), port=args.port)

        print("Loop is over")

        # Send shutdown signal to main thread
        streaming_request_queue.put(None)
    else:
        # Other ranks participate in the chat server by waiting
        worker_loop()

    dist.destroy_process_group()


# Run the app
if __name__ == "__main__":
    main()
