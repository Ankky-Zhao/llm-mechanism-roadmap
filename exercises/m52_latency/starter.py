import inspect
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def model_step(model, input_ids, cache=None):
    """Run the model and return (logits, cache). Nothing to change here.

    Hugging Face renamed the argument from `past_key_values` to `cache`; this works with both.
    """
    name = "cache" if "cache" in inspect.signature(model.forward).parameters else "past_key_values"
    out = model(input_ids, use_cache=True, **{name: cache})
    return out.logits, out[name]


def cache_tensors(cache):
    """The list of (keys, values) tensors, one pair per layer. Nothing to change here."""
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return [(layer[0], layer[1]) for layer in cache]


@torch.no_grad()
def generate_no_cache(model, input_ids, max_new_tokens):
    """Greedy decoding that runs the full sequence through the model at every step.

    Returns (ids (1, T + max_new_tokens), times), where times[i] is the wall-clock time in seconds
    spent producing new token i. Measure with time.perf_counter().
    """
    # TODO 1
    raise NotImplementedError


@torch.no_grad()
def generate_with_cache(model, input_ids, max_new_tokens):
    """Greedy decoding with the KV cache.

    Prefill: one call of model_step on the whole prompt gives the first new token and the cache.
    Decode:  every later call receives only the newest token, shape (1, 1), plus the cache.
    Returns (ids, times, cache), with ids and times as in generate_no_cache.
    """
    # TODO 2
    raise NotImplementedError


def latency_summary(times):
    """The two numbers serving systems report.

    ttft:              time to first token, times[0]
    tokens_per_second: decoding speed after the first token, (number of later tokens) / (their total time)
    Returns a dict with these two keys.
    """
    # TODO 3
    raise NotImplementedError


def kv_cache_bytes(n_layer, n_head, d_head, seq_len, bytes_per_number=4, batch=1):
    """Memory of the cache: one key and one value vector per layer, head and position."""
    # TODO 4
    raise NotImplementedError
