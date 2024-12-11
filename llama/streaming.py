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
from transformers import TextStreamer
import time
from torch import distributed as dist


class MasterRankTextStreamer(TextStreamer):
    """
    Text streamer that only prints text from the master rank.
    For more information, see :class:`~transformers.TextStreamer`.
    """

    def __init__(self, tokenizer, skip_prompt=True, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.time_per_token = []
        self.last_time = time.time()
        self.last_message = ""
        self._prev_token = None

    def clear_last_message(self):
        self.last_message = ""

    def put(self, value):
        if dist.get_rank() != 0:
            return
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

        time_since_last_token = time.time() - self.last_time
        v = self.tokenizer.batch_decode(value, skip_special_tokens=True)
        self.last_message += v[0]
        print(v[0], end="", flush=True)
        self.time_per_token.append(time_since_last_token)
        self.last_time = time.time()

    def on_finalized_text(self, _: str, stream_end: bool = False):
        if stream_end:
            self._prev_token = None
