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
    times = []
    for _ in range(max_new_tokens):
        t0 = time.perf_counter()
        next_id = model(input_ids).logits[0, -1].argmax()
        input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)
        times.append(time.perf_counter() - t0)
    return input_ids, times


@torch.no_grad()
def generate_with_cache(model, input_ids, max_new_tokens):
    times = []
    t0 = time.perf_counter()
    logits, cache = model_step(model, input_ids)                 # prefill: the whole prompt in one pass
    next_id = logits[0, -1].argmax()
    input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)
    times.append(time.perf_counter() - t0)
    for _ in range(max_new_tokens - 1):
        t0 = time.perf_counter()
        logits, cache = model_step(model, next_id.view(1, 1), cache)   # decode: only the newest token
        next_id = logits[0, -1].argmax()
        input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)
        times.append(time.perf_counter() - t0)
    return input_ids, times, cache


def latency_summary(times):
    ttft = times[0]
    decode = times[1:]
    return {"ttft": ttft, "tokens_per_second": len(decode) / sum(decode)}


def kv_cache_bytes(n_layer, n_head, d_head, seq_len, bytes_per_number=4, batch=1):
    return 2 * n_layer * batch * n_head * seq_len * d_head * bytes_per_number
