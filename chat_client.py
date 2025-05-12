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
"""
A simple streaming chat client based on the openai library.
"""

import argparse
import atexit
import json
import os
import readline

import openai


def chat_loop(model: str, url: str, args):
    conversation = []
    client = openai.OpenAI(api_key="None", base_url=url)
    temperature = openai.NOT_GIVEN
    total_tokens = None
    if args.total_tokens is not None:
        total_tokens = args.total_tokens
    system_prompt = args.system_prompt
    if system_prompt is not None:
        conversation.append({"role": "system", "content": system_prompt})

    print(
        "Type a message to start the chat.",
        "Press ctrl-D or type '/exit' to end the conversation.",
        "Type '/clear' to clear the chat context.",
        "Type '/help' to see the list of available commands.",
        f"Commands are stored in history file {args.history}.",
    )

    if args.load:
        with open(args.load, "r") as fp:
            conversation = json.load(fp)
        print(f"Loaded history from {args.load}")

    try:
        readline.read_history_file(args.history)
        readline.set_history_length(1000)
    except FileNotFoundError:
        pass

    atexit.register(readline.write_history_file, args.history)
    tokens_so_far = 0
    try:
        while True:
            skip = False
            message = input("> ")
            # Commands
            if message.strip().startswith("/"):
                command = message[1:].strip()
                if command == "help":
                    print(
                        "Commands:\n",
                        "  /help: Show this help message\n",
                        "  /exit: End the conversation\n",
                        "  /clear: Clear the chat context\n",
                        "  /temp <float>: Set the temperature for the model\n",
                        "  /tokens <int>: Set the maximal number of tokens to use per response\n",
                        "  /totaltokens <int>: Set the maximal number of tokens to use in total\n",
                        "  /cq <prompt>: Prefix <prompt> with custom prompt given by --custom-prompt <FILE>\n",
                        "  /save [file.json]: Save current chat history (Default: chat_history.json)\n",
                        "  /load <str>: Load chat history from the given file\n",
                        "  /tool <str>: Responds as \"tool\" instead of \"user\"\n",
                    )
                    continue
                elif command == "exit":
                    raise EOFError
                elif command == "clear":
                    print("[Chat context cleared]")
                    conversation = []
                    tokens_so_far = 0
                    continue
                elif command.startswith("temp "):
                    try:
                        temperature = float(command.split(" ")[1])
                        if temperature <= 0 or temperature > 1:
                            raise ValueError
                        print(f"[Temperature set to {temperature}]")
                    except ValueError:
                        print("[Invalid temperature. Should be a positive number less than 1]")
                    continue
                elif command.startswith("tokens "):
                    try:
                        tokens = int(command.split(" ")[1])
                        if tokens < 0 or tokens > 131000:
                            raise ValueError
                        print(f"[Tokens set to {tokens}]")
                        args.max_tokens = tokens
                    except ValueError:
                        print("[Invalid number of tokens. Should be a positive number less than 131,000]")
                    continue
                elif command.startswith("totaltokens "):
                    try:
                        tokens = int(command.split(" ")[1])
                        if tokens < 0 or tokens > 131000:
                            raise ValueError
                        print(f"[Total Tokens set to {tokens}]")
                        total_tokens = tokens
                    except ValueError:
                        print("[Invalid number of tokens. Should be a positive number less than 131,000]")
                    continue
                elif command.startswith("sofar"):
                    print("[Tokens in conversation:", tokens_so_far, "]")
                    continue
                elif command.startswith("cq "):
                    if args.custom_prompt is None:
                        print("[Error: a custom prompt has not been provided, use --custom-prompt]")
                        continue
                    message = command[len("cq ") :]
                    new_message = open(args.custom_prompt, "r").read()
                    new_message += message + "]"
                    message = new_message
                elif command.startswith("save"):
                    filename = None
                    if command == "save":
                        filename = "chat_history.json"
                    elif command.startswith("save "):
                        filename = command[len("save ") :]
                    if not filename:
                        print("[Error: invalid syntax for /save]")
                        continue
                    with open(filename, "w") as fp:
                        json.dump(conversation, fp)
                    print(f"[Saved history to {filename}]")
                    continue
                elif command.startswith("load "):
                    filename = command[len("load ") :]
                    if not os.path.exists(filename):
                        print(f'Cannot load "{filename}", file does not exist')
                        continue
                    with open(filename, "r") as fp:
                        conversation = json.load(fp)
                    print(f"[Loaded history from {filename}]")
                    continue
                elif command.startswith("tool "):
                    response = command[len("tool ") :]
                    conversation.append({"role": "tool", "content": response})
                    skip = True
                else:
                    print(f"Invalid command '{command}'")
                    continue

            if not skip:
                conversation.append({"role": "user", "content": message})

            try:
                max_tokens = args.max_tokens
                if total_tokens is not None:
                    max_tokens = total_tokens - tokens_so_far

                print(conversation)
                chat_completion = client.chat.completions.create(
                    model=model,
                    messages=conversation,
                    stream=not args.no_stream,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    # tools=tools,
                )

                if args.no_stream:
                    response = chat_completion.choices[0].message.content
                    response = response.replace("\\n", "\n")
                    print(response)
                    conversation.append({"role": "assistant", "content": response})
                else:
                    response_to_save = ""
                    full_response = ""
                    in_reasoning = False
                    for chunk in chat_completion:
                        if chunk.choices[0].delta.content is not None:
                            if chunk.choices[0].delta.content == "<think>":
                                print("\n==========================REASONING===========================")
                                in_reasoning = True
                                continue
                            if chunk.choices[0].delta.content == "</think>":
                                print("\n==========================END REASONING===========================")
                                in_reasoning = False
                                continue

                            if not in_reasoning:
                                response_to_save += chunk.choices[0].delta.content
                                full_response += chunk.choices[0].delta.content
                                tokens_so_far += 1
                            else:
                                full_response += chunk.choices[0].delta.content
                            print(chunk.choices[0].delta.content, end="", flush=True)
                            if args.stop_after and response_to_save.endswith(args.stop_after):
                                print("\n[Stopping after seeing the stop pattern]")
                                break
                    print()

                    full_response += "\n"
                    response_message = {"role": "assistant", "content": response_to_save}
                    conversation.append(response_message)
            except KeyboardInterrupt:  # Catch ctrl-C
                if not args.no_stream:
                    chat_completion.close()
                print("\n[Response interrupted]")
    except EOFError:
        print("[Ending chat]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="LLLama")
    parser.add_argument("--url", type=str, default="http://localhost:8123")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--total-tokens", type=int, default=None)
    parser.add_argument("--custom-prompt", type=str, default=None)
    parser.add_argument("--system-prompt", type=str, default=None)
    parser.add_argument("--history", action="store", type=str, default=".chat-client-history")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument(
        "--load", type=str, default=None, help="Continue from a file containing an existing chat history"
    )
    parser.add_argument("--stop-after", type=str, default=None, help="Checkpoint chat when seeing the given pattern")

    args = parser.parse_args()
    chat_loop(args.model, args.url, args)


if __name__ == "__main__":
    main()
