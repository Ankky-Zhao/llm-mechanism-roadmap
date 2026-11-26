# Module 0.2 counted the parameters of GPT-2. Most current open-weight models use the Llama-style
# block instead: RMSNorm, SwiGLU, grouped-query attention, no bias terms, no position table (RoPE
# has no parameters). This exercise counts such a model from its config alone.


def llama_param_count(cfg):
    """Number of parameters of a Llama-style decoder, from a config dict with the keys
    vocab_size, hidden_size, intermediate_size, num_hidden_layers,
    num_attention_heads, num_key_value_heads, tie_word_embeddings.

    Per block (no biases anywhere):
      attention  q: d x d      k and v: d x d_kv each, with d_kv = (d / n_heads) * n_kv_heads      output: d x d
      SwiGLU     three matrices: d x m, d x m, m x d                (m = intermediate_size)
      RMSNorm    two weight vectors of length d
    Outside the blocks: the embedding V x d, one final RMSNorm (d), and an output matrix V x d
    only if tie_word_embeddings is False.
    """
    # TODO 1
    raise NotImplementedError


def model_bytes(n_params, bits_per_weight):
    """Memory of the weights in bytes."""
    # TODO 2
    raise NotImplementedError


def precisions_that_fit(n_params, memory_gb, overhead=1.2, options=(16, 8, 4)):
    """Which of the bit widths in `options` fit into memory_gb gigabytes (1 GB = 1e9 bytes)?

    Count the weights times `overhead`, a rough allowance for activations and the KV cache.
    Returns the fitting bit widths as a list, in the order given.
    """
    # TODO 3
    raise NotImplementedError
