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

    def clear_last_message(self):
        self.last_message = ""

    def put(self, value):
        if dist.get_rank() != 0:
            return
        if self.skip_prompt and self.next_tokens_are_prompt:
            # Skip both the prompt and the content header (in LLaMA 3.1, the
            # sequence separator is 128007)
            if (
                value.shape[-1] == 1 and value.item() == 128007
            ):  # Skip until start of answer
                self.next_tokens_are_prompt = False
            return

        time_since_last_token = time.time() - self.last_time
        v = self.tokenizer.batch_decode(value, skip_special_tokens=True)
        self.last_message += v[0]
        print(v[0], end="", flush=True)
        self.time_per_token.append(time_since_last_token)
        self.last_time = time.time()
