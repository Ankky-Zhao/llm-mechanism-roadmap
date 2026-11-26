import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def _ln(x, ln):
    return F.layer_norm(x, (x.shape[-1],), ln.weight, ln.bias, ln.eps)


@torch.no_grad()
def gpt2_block(x, block):
    x = x + gpt2_attention(_ln(x, block.ln_1), block.attn)     # read from other positions, add to the stream
    x = x + gpt2_mlp(_ln(x, block.ln_2), block.mlp)            # process each position, add to the stream
    return x


@torch.no_grad()
def gpt2_logits(model, input_ids):
    t = model.transformer
    T = input_ids.shape[1]
    x = t.wte.weight[input_ids] + t.wpe.weight[torch.arange(T)]
    for block in t.h:
        x = gpt2_block(x, block)
    return _ln(x, t.ln_f) @ t.wte.weight.T
