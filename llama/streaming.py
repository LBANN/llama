from transformers import TextStreamer
import time
from torch import distributed as dist


class MasterRankTextStreamer(TextStreamer):
    """
    Text streamer that only prints text from the master rank.
    For more information, see :class:`~transformers.TextStreamer`.
    """

    def __init__(self, tokenizer, skip_prompt=False, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.time_per_token = []
        self.last_time = time.time()

    def put(self, value):
        if dist.get_rank() != 0:
            return
        time_since_last_token = time.time() - self.last_time
        v = self.tokenizer.batch_decode(value, skip_special_tokens=True)
        print(v[0], end='', flush=True)
        self.time_per_token.append(time_since_last_token)
        self.last_time = time.time()
