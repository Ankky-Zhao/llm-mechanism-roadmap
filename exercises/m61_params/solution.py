def llama_param_count(cfg):
    V, d, m, L = cfg["vocab_size"], cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
    d_head = d // cfg["num_attention_heads"]
    d_kv = d_head * cfg["num_key_value_heads"]              # grouped-query attention: fewer key/value heads
    attention = d * d + 2 * d * d_kv + d * d                # q, (k, v), output projection; no biases
    mlp = 3 * d * m                                         # SwiGLU: gate, up, down
    norms = 2 * d                                           # two RMSNorm weight vectors per block
    total = V * d + L * (attention + mlp + norms) + d       # embedding, blocks, final norm
    if not cfg["tie_word_embeddings"]:
        total += V * d                                      # a separate output matrix
    return total


def model_bytes(n_params, bits_per_weight):
    return n_params * bits_per_weight / 8


def precisions_that_fit(n_params, memory_gb, overhead=1.2, options=(16, 8, 4)):
    return [b for b in options if model_bytes(n_params, b) * overhead <= memory_gb * 1e9]
