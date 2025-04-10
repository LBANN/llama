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

import base64
import asyncio
import atexit
import json
import os
import queue
import sys
import threading
from io import BytesIO
import time
from dataclasses import dataclass
from PIL import Image
from urllib.parse import urlparse

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
from transformers import MllamaForConditionalGeneration, AutoProcessor, \
    Llama4ForConditionalGeneration, TextIteratorStreamer

# Restore affinity
Process().cpu_affinity(affinity)

# Create a FastAPI app
app = FastAPI()

# Global variables
model = tokenizer = None
processor = None
device = torch.device("cuda:0")
max_batch_size = None
request_queue: queue.Queue = None

MAX_TOKENS_DEFAULT = 512
TEMPERATURE_DEFAULT = 1.0
TOP_P_DEFAULT = 1.0


@dataclass
class ChatRequest:
    """
    Object that stores a chat request.
    """

    inputs: list[int]
    max_tokens: int
    settings: dict[str, float]
    response_queue: queue.Queue
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
            if value[i].item() == self.tokenizer.eos_token_id:
                q.put(None)
            else:
                q.put(str(v[i]))

    def on_finalized_text(self, _: str, stream_end: bool = False):
        if stream_end:
            self._prev_token = None


def load_image_from_url(url):
    try:
        url_spec = urlparse(url)

        if url_spec.scheme.startswith("http"):
            msg = "Only base64 data URLs are supported for now."
            raise NotImplementedError(msg)

        if url_spec.scheme == "data":
            data_spec, data = url_spec.path.split(",", 1)
            media_type, data_type = data_spec.split(";", 1)

            if data_type != "base64":
                msg = "Only base64 data URLs are supported for now."
                raise NotImplementedError(msg)
            image = Image.open(BytesIO(base64.b64decode(data)))
            image.load()
            return image.convert("RGB")
            # return media_io.load_base64(media_type, data)

        if url_spec.scheme == "file":
            msg = "Only base64 data URLs are supported for now."
            raise NotImplementedError(msg)

        msg = "The URL must be either a HTTP, data or file URL."
        raise ValueError(msg)
    except Exception as e:
        print('could not load image:', e)


# Define a route for OpenAI API compatibility
@app.post("/chat/completions")
async def completions(request: Request):
    # Read the request body as JSON
    request_body = await request.json()

    # Handle the request
    messages = request_body.get("messages", [])
    if "max_tokens" in request_body:
        max_tokens = request_body.get(
            "max_tokens", MAX_TOKENS_DEFAULT,
        )
    if "max_completion_tokens" in request_body:
        # openai deprciated max_tokens
        max_tokens = request_body.get(
            "max_completion_tokens", MAX_TOKENS_DEFAULT,
        )
    stream = request_body.get("stream", False)
    settings = {}
    # print(type(messages), messages)
    if "temperature" in request_body:
        settings["temperature"] = request_body.get(
            "temperature", TEMPERATURE_DEFAULT,
        )
    if "top_p" in request_body:
        settings["top_p"] = request_body.get(
            "top_p", TOP_P_DEFAULT,
        )

    if tokenizer is not None:
        actual_inputs: torch.Tensor = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
        )
        inputs = actual_inputs.flatten().tolist()
    else:
        images = []
        # search if there is images
        for message in messages:
            if 'content' in message:
                for cont in message['content']:
                    try:
                        if isinstance(cont, dict):
                            if 'image_url' in cont.keys():
                                image = load_image_from_url(cont['image_url']["url"])
                                images.append(image)
                    except Exception as e:
                        print('could not load image:', e)

        actual_inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True,
        )
        # print('Actual inputs\n', actual_inputs)
        if len(images) == 0:
            inputs = processor(
                text=actual_inputs, return_tensors="pt",
            ).to(device)
        else:
            # TODO: You need to broadcast inputs['pixel_values'] from rank 0 to
            # all other devices. These pixel values then need to be fed into
            # the model.generate calls.
            print('Images not yet supported.')
            inputs = processor(
                text=actual_inputs, return_tensors="pt",
                # images=images,
            ).to(device)
        # print('inputs\n', inputs)
        inputs = inputs['input_ids'].flatten().tolist()

    input_len = len(inputs)

    response_queue = queue.Queue()
    message_queue = queue.Queue()
    chat_request = ChatRequest(
        inputs, max_tokens, settings, response_queue, message_queue
    )

    request_queue.put(chat_request)

    async def get_tokens(request: Request):
        try:
            while True:
                await asyncio.sleep(0.001)  # Yield execution

                # Check if the client has disconnected
                if await request.is_disconnected():
                    print("Client has disconnected")
                    message_queue.put(None)
                    response_queue.get()  # Get final signal
                    break
                try:
                    res = response_queue.get(block=False)
                    if res is None:
                        break
                    yield res
                except queue.Empty:
                    pass

        except asyncio.CancelledError:
            print("Chat was interrupted")
            message_queue.put(None)
            response_queue.get()  # Get final signal

    # Return a streaming response
    if stream:

        async def content_stream(request):
            async for token in get_tokens(request):
                response = {
                    "choices": [{"delta": {"role": "assistant", "content": token}}],
                }
                yield f"data: {json.dumps(response)}\n\n"

        return StreamingResponse(
            content=content_stream(request), media_type="text/event-stream"
        )
    else:
        outputs = [token async for token in get_tokens(request)]
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


def aggregate_tasks(request_queue: queue.Queue, max_batch_size: int):
    """
    Aggregate tasks from the request queue and process them in a batch.
    """
    requests: list[ChatRequest] = []
    # Process up to `max_batch_size` requests
    print("Queue size:", request_queue.qsize())
    while request_queue.qsize() > 0:
        requests.append(request_queue.get())

        # Check for shutdown signal
        if requests[-1] is None:
            return None

        if len(requests) >= max_batch_size:
            break

    # Aggregate the inputs and settings
    qlen = len(requests)
    input_len = max([len(x.inputs) for x in requests])
    if tokenizer is not None:
        actual_inputs = torch.full(
            (qlen, input_len), tokenizer.eos_token_id, device=device,
            dtype=torch.long,
        )
    else:
        actual_inputs = torch.full(
            (qlen, input_len), processor.tokenizer.eos_token_id, device=device,
            dtype=torch.long,
        )
    for i, request in enumerate(requests):
        # print(i, request)
        actual_inputs[i, -len(request.inputs) :] = torch.tensor(request.inputs)

    max_tokens = max([x.max_tokens for x in requests])
    settings = {}
    for request in requests:
        settings.update(request.settings)

    message_queues = [x.message_queue for x in requests]
    response_queues = [x.response_queue for x in requests]
    input_lengths = [len(x.inputs) for x in requests]

    return (
        actual_inputs,
        max_tokens,
        settings,
        message_queues,
        response_queues,
        input_lengths,
    )


def master_loop(
    inputs,
    device,
    request_queue,
    batch_delay,
    interval_minutes=5,
    model_ver=3,
):
    if model_ver == 3:
        cache_manager = KVCacheManager(model)
    last_sync_time = time.time()
    while True:
        try:
            # Aggregate tasks from the request queues
            if not request_queue.empty():
                task = aggregate_tasks(request_queue, max_batch_size)
            else:
                # No tasks
                if time.time() - last_sync_time > interval_minutes * 60:
                    # Send a keepalive signal if too much time has passed
                    chat_synchronize_ranks(
                        device, ControlInfo(message=ControlMessageType.KEEPALIVE)
                    )
                    last_sync_time = time.time()

                # Otherwise, wait
                time.sleep(batch_delay / 1000)
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
                response_queues,
                input_lengths,
            ) = task
            input_len = inputs.shape[1]

            # Synchronize the input tokens and lengths
            control_info = ControlInfo(
                batch_size=inputs.shape[0],
                input_len=inputs.shape[1],
                max_new_tokens=max_tokens,
                temperature=settings.get("temperature", None),
                top_p=settings.get("top_p", None),
            )
            chat_synchronize_ranks(device, control_info)
            last_sync_time = time.time()
            kwargs = control_info.to_kwargs()
            dist.broadcast(inputs, 0)
            attention_mask = torch.ones_like(inputs)
            for i, length in enumerate(input_lengths):
                if length < input_len:
                    attention_mask[i, 0 : input_len - length] = 0
            dist.broadcast(attention_mask, 0)
            if model_ver == 3:
                # Prepare attention mask based on input lengths

                streamer = ChatServerTextStreamer(
                    tokenizer, response_queues, message_queues
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

            elif model_ver == 4:

                streamer = TextIteratorStreamer(
                    processor, skip_prompt=True, skip_special_tokens=True,
                )
                streamer_inputs = {
                    'input_ids': inputs,
                    'attention_mask': attention_mask,
                }
                # TODO: support grabbing settings...
                # print('streamer inputs', streamer_inputs)
                # keywords = dict(streamer_inputs, **settings)
                keywords = dict(streamer_inputs, **kwargs)
                keywords['streamer'] = streamer
                keywords['max_new_tokens'] = max_tokens

                # keywords['cache_implementation'] = "hybrid"
                # create a thread to pull results from the model on one thread
                thread = threading.Thread(
                    target=model.generate, kwargs=keywords,
                )
                thread.start()
                response = ""
                for new_text in streamer:
                    # put the new_text into a queue that is going back to a streaming client
                    response_queues[0].put(new_text)
                    response += new_text
                    # print(response)
                thread.join()
                print(f'[rank:{dist.get_rank()}] done with streaming')

            # Send signal to end the stream
            for q in response_queues:
                q.put(None)
        except queue.Empty:
            # Send a keepalive signal
            chat_synchronize_ranks(
                device, ControlInfo(message=ControlMessageType.KEEPALIVE)
            )
            last_sync_time = time.time()
        except StopIteration:  # Chat interrupted
            # Clear KV cache on interruption
            if model_ver == 3:
                cache_manager.clear()

            # Send signal to end the stream
            for q in response_queues:
                q.put(None)


def worker_loop(model_ver=3):
    if model_ver == 3:
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
                if model_ver == 3:
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
                else:
                    streamer = TextIteratorStreamer(
                        processor, skip_prompt=True, skip_special_tokens=True,
                    )
                    # CJ TODO: Do i need to grab the settings...
                    streamer_inputs = {
                        'input_ids': inputs,
                        'attention_mask': attention_mask,
                    }
                    # print('streamer inputs', streamer_inputs)
                    keywords = dict(streamer_inputs, **kwargs)
                    keywords['streamer'] = streamer
                    keywords['max_new_tokens'] = info.max_new_tokens
                    # keywords['cache_implementation'] = "hybrid"
                    # create a thread to pull results from the model
                    thread = threading.Thread(
                        target=model.generate, kwargs=keywords,
                    )
                    thread.start()
                    response = ""
                    for new_text in streamer:
                        response += new_text
                    thread.join()
                    print(f'[rank:{dist.get_rank()}] done with streaming')
            except StopIteration as ex:  # Chat interrupted
                info = ex.value
                if info is not None and info.message == ControlMessageType.EXIT:
                    break
                elif info is not None and info.message == ControlMessageType.CANCEL:
                    # Clear KV cache on interruption
                    if model_ver == 3:
                        cache_manager.clear()

        info = chat_synchronize_ranks(device)


def main(running_under_server=False):
    args = get_args(server=not running_under_server)

    if not dist.is_initialized():
        dist.init_process_group("nccl")
    global model
    global tokenizer
    global processor
    global inputs
    global max_batch_size

    if args.model_ver == 3:
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

        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left")
        model = DistributedLlama(
            args.model_dir,
            device,
            device_mesh,
            delay_init=True,
            load_checkpoint=not args.benchmark,
            io_threads=io_threads,
        )
        barrier(device)

        inputs = torch.full((1, 131072), 128002, dtype=torch.long, device=device)

        max_batch_size = args.max_batch_size

    elif args.model_ver == 4:
        global TEMPERATURE_DEFAULT
        TEMPERATURE_DEFAULT = 0.6
        global TOP_P_DEFAULT
        TOP_P_DEFAULT = 0.9

        if 'Llama-4' in args.model_dir:
            print('Trying to load llama-4')
            model = Llama4ForConditionalGeneration.from_pretrained(
                args.model_dir,
                tp_plan='auto',
                torch_dtype='auto',
                # attn_implementation="flex_attention",  # crashes after a cache miss
                attn_implementation="eager",
            )
        else:
            print('Trying to load llama-3.2')
            model = MllamaForConditionalGeneration.from_pretrained(
                args.model_dir,
                tp_plan='auto',
                torch_dtype='auto',
            )
        print(model.dtype)
        processor = AutoProcessor.from_pretrained(
            args.model_dir
        )
        barrier(device)

        inputs = torch.full(
            (1, 200002),  # EOS TOKEN ID
            10485760,  # max positional embeddings
            dtype=torch.long, device=device)

        max_batch_size = args.max_batch_size
    else:
        raise NotImplementedError(
            f"Llama model version {args.model_ver} is not supported yet",
        )

    barrier(device)

    if args.compile:
        model.model.forward = torch.compile(model.model.forward)

        if args.static_cache_size > 0:
            model.model.original_forward = model.model.forward
            model.model.static_cache_forward = torch.compile(
                model.model.forward, mode="reduce-overhead", dynamic=True
            )
            model.static_cache_size = args.static_cache_size

    # Initialize the model serving thread loop
    if dist.get_rank() == 0:
        # Create request queues for users
        global request_queue
        request_queue = queue.Queue()

        # Start the keepalive thread
        gen_thread = threading.Thread(
            target=master_loop,
            args=(
                inputs,
                device,
                request_queue,
                args.batch_delay,
            ),
            kwargs={'model_ver': args.model_ver},
            daemon=True,
        )
        gen_thread.start()

        # Run the uvicorn server if necessary
        if not running_under_server:
            # Detect the hostname and print it
            import socket

            print("Running server on", socket.gethostname())

            uvicorn.run(app, host=socket.gethostname(), port=args.port)

            print("Loop is over")

            # Send shutdown signal to main thread
            request_queue.put(None)
            dist.destroy_process_group()
        else:
            atexit.register(dist.destroy_process_group)
    else:
        # Other ranks participate in the chat server by waiting
        worker_loop(model_ver=args.model_ver)
        dist.destroy_process_group()


# Run the app
if __name__ == "__main__":
    main()
elif os.getenv("SERVER_SOFTWARE", "").startswith("gunicorn"):
    # Gunicorn server
    if sys.argv.index("--") > 0:
        sys.argv = sys.argv[sys.argv.index("--") + 1 :]
    sys.argv.insert(0, __name__)

    # Initialize the server thread
    main(running_under_server=True)

elif sys.argv[0].endswith("uvicorn") or Process().parent().name() == "uvicorn":
    raise RuntimeError(
        "Running under uvicorn is not supported, please run the script directly or use gunicorn."
    )
