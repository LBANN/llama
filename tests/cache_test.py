from dataclasses import dataclass
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from llama.chat_utils import KVCacheManager
import pytest


@dataclass
class MockLlama:
    model: LlamaForCausalLM


@pytest.mark.parametrize("static_cache", [True, False])
def test_varying_batch_size(static_cache):
    config = LlamaConfig(
        hidden_size=1024,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=2048,
    )
    model = LlamaForCausalLM(config)
    batch_sizes = [1, 8, 8, 5]
    input_len = 16
    max_tokens = 16
    inputs = [
        torch.randint(0, 100, (batch_size, input_len)) for batch_size in batch_sizes
    ]
    attn_masks = [torch.ones((batch_size, input_len)) for batch_size in batch_sizes]

    lllama = MockLlama(model)
    if static_cache:
        lllama.model.original_forward = lllama.model.forward
        lllama.model.static_cache_forward = lllama.model.forward
        lllama.static_cache_size = input_len + max_tokens
    cache_mgr = KVCacheManager(lllama)

    for input, attn_mask in zip(inputs, attn_masks):
        outputs_without_cache = model.generate(
            input_ids=input,
            attention_mask=attn_mask,
            max_new_tokens=max_tokens,
        )

        outputs_with_cache = model.generate(
            input_ids=input,
            attention_mask=attn_mask,
            max_new_tokens=max_tokens,
            past_key_values=cache_mgr.get_cache(input, input_len, max_tokens),
        )
        cache_mgr.update(outputs_with_cache)

        assert torch.equal(outputs_without_cache, outputs_with_cache)


if __name__ == "__main__":
    test_varying_batch_size(False)
    test_varying_batch_size(True)
