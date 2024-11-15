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
import openai


def chat_loop(model: str, url: str):
    conversation = []
    client = openai.OpenAI(api_key="None", base_url=url)

    print(
        "Type a message to start the chat.",
        "Press ctrl-D or type 'exit' to end the conversation.",
        "Type 'clear' to clear the chat context.",
    )

    try:
        while True:
            message = input("> ")
            if message.strip() == "exit":
                raise EOFError
            if message.strip() == "clear":
                print("[Chat context cleared]")
                conversation = []
                continue
            conversation.append({"role": "user", "content": message})

            chat_completion = client.chat.completions.create(
                model=model, messages=conversation, stream=True
            )
            full_response = ""
            for chunk in chat_completion:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    print(chunk.choices[0].delta.content, end="", flush=True)

            print()
            full_response += "\n"
            response_message = {"role": "assistant", "content": full_response}
            conversation.append(response_message)
    except EOFError:
        print("[Ending chat]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="LLLama")
    parser.add_argument("--url", type=str, default="http://localhost:8123")

    args = parser.parse_args()
    chat_loop(args.model, args.url)


if __name__ == "__main__":
    main()
