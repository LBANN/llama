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
import os
import openai
import readline


def chat_loop(model: str, url: str, args):
    conversation = []
    client = openai.OpenAI(api_key="None", base_url=url)
    temperature = openai.NOT_GIVEN

    print(
        "Type a message to start the chat.",
        "Press ctrl-D or type '/exit' to end the conversation.",
        "Type '/clear' to clear the chat context.",
        "Type '/help' to see the list of available commands.",
        f"Commands are stored in history file {args.history}.",
    )

    try:
        readline.read_history_file(args.history)
        readline.set_history_length(1000)
    except FileNotFoundError:
        pass

    atexit.register(readline.write_history_file, args.history)

    try:
        while True:
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
                        "  /cq <prompt>: Prefix <prompt> with custom prompt given by --custom-prompt <FILE>",
                    )
                    continue
                elif command == "exit":
                    raise EOFError
                elif command == "clear":
                    print("[Chat context cleared]")
                    conversation = []
                    continue
                elif command.startswith("temp "):
                    try:
                        temperature = float(command.split(" ")[1])
                        if temperature <= 0 or temperature > 1:
                            raise ValueError
                        print(f"[Temperature set to {temperature}]")
                    except ValueError:
                        print(
                            "[Invalid temperature. Should be a positive number less than 1]"
                        )
                    continue
                elif command.startswith("tokens "):
                    try:
                        tokens = int(command.split(" ")[1])
                        if tokens < 0 or tokens > 131000:
                            raise ValueError
                        print(f"[Tokens set to {tokens}]")
                        args.max_tokens = tokens
                    except ValueError:
                        print(
                            "[Invalid number of tokens. Should be a positive number less than 131,000]"
                        )
                    continue
                elif command.startswith("cq "):
                    if args.custom_prompt is None:
                        print(
                            "[Error: a custom prompt has not been provided, use --custom-prompt]"
                        )
                        continue
                    message = command[len("cq ") :]
                    new_message = open(args.custom_prompt, "r").read()
                    new_message += message + "]"
                    message = new_message
                else:
                    print(f"Invalid command '{command}'")
                    continue

            conversation.append({"role": "user", "content": message})

            chat_completion = client.chat.completions.create(
                model=model,
                messages=conversation,
                stream=not args.no_stream,
                temperature=temperature,
                max_tokens=args.max_tokens,
            )
            if args.no_stream:
                response = chat_completion.choices[0].message.content
                response = response.replace("\\n", "\n")
                print(response)
            else:
                full_response = ""
                try:
                    for chunk in chat_completion:
                        if chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content
                            print(chunk.choices[0].delta.content, end="", flush=True)
                    print()
                except KeyboardInterrupt:  # Catch ctrl-C
                    chat_completion.close()
                    print("\n[Response interrupted]")

                full_response += "\n"
                response_message = {"role": "assistant", "content": full_response}
                conversation.append(response_message)
    except EOFError:
        print("[Ending chat]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="LLLama")
    parser.add_argument("--url", type=str, default="http://localhost:8123")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--custom-prompt", type=str, default=None)
    parser.add_argument(
        "--history", action="store", type=str, default=".chat-client-history"
    )
    parser.add_argument("--no-stream", action="store_true")

    args = parser.parse_args()
    chat_loop(args.model, args.url, args)


if __name__ == "__main__":
    main()
