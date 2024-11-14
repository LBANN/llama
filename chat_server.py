from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json
import uvicorn

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, TextStreamer

from llama import DistributedLlama, LlamaDeviceMesh
from llama.chat_utils import get_args, barrier, chat_synchronize_ranks

# Create a FastAPI app
app = FastAPI()

# Global variables
input_len = inputs = model = tokenizer = None
device = torch.device("cuda:0")


class ChatServerTextStreamer(TextStreamer):
    """
    Text streamer that interacts with a streaming response for a chat server.
    """

    def __init__(self, tokenizer, skip_prompt=True, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)

    def put(self, value):
        if self.skip_prompt and self.next_tokens_are_prompt:
            # Skip both the prompt and the content header (in LLaMA 3.1, the
            # sequence separator is 128007)
            if (
                value.shape[-1] == 1 and value.item() == 128007
            ):  # Skip until start of answer
                self.next_tokens_are_prompt = False
            return

        v = self.tokenizer.batch_decode(value, skip_special_tokens=True)

        response = {"choices": [{"delta": {"role": "assistant", "content": str(v[0])}}]}
        yield f"data: {json.dumps(response)}\n\n"


# Define a route for OpenAI API compatibility
@app.post("/chat/completions")
async def completions(request: Request):
    # Read the request body as JSON
    request_body = await request.json()

    # Handle the request
    # print(request_body)
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

    # Synchronize the input tokens and lengths
    chat_synchronize_ranks(inputs, input_len, device)

    # Generate text as a streaming response
    def generate_text():
        yield from model.generate(
            input_ids=inputs[:, : input_len[0]],
            attention_mask=torch.ones((1, input_len[0]), device=device),
            streamer=ChatServerTextStreamer(tokenizer),
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Return a streaming response
    if stream:
        return StreamingResponse(
            content=generate_text(), media_type="text/event-stream"
        )
    else:
        raise NotImplementedError("Non-streaming completions are not supported")


def event_loop():
    chat_synchronize_ranks(inputs, input_len, device)
    while input_len[0] > 0:
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
        # Detect the hostname and print it
        import socket

        print("Running server on", socket.gethostname())

        uvicorn.run(app, host="0.0.0.0", port=args.port)

        print("Loop is over")

        # Tear down the process group
        input_len[0] = 0
        chat_synchronize_ranks(inputs, input_len, device)
    else:
        # Other ranks participate in the chat server by waiting
        event_loop()


# Run the app
if __name__ == "__main__":
    main()
