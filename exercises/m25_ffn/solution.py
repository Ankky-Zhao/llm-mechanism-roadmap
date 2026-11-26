import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


@torch.no_grad()
def gpt2_mlp(x, mlp):
    h = F.gelu(x @ mlp.c_fc.weight + mlp.c_fc.bias, approximate="tanh")     # 768 -> 3072
    return h @ mlp.c_proj.weight + mlp.c_proj.bias                          # 3072 -> 768


def swiglu_ffn(x, w_gate, w_up, w_down):
    g = x @ w_gate
    gate = g * torch.sigmoid(g)                 # SiLU, also called Swish
    return (gate * (x @ w_up)) @ w_down


def ffn_param_count(d, hidden, kind):
    return (2 if kind == "mlp" else 3) * d * hidden


def swiglu_hidden_for_same_params(d):
    return 8 * d // 3
