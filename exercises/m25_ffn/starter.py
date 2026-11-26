import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

# GPT-2's MLP is  mlp = model.transformer.h[i].mlp  with two Conv1D layers
#   mlp.c_fc    768 -> 3072     (y = x @ weight + bias)
#   mlp.c_proj  3072 -> 768
# and the activation F.gelu(..., approximate="tanh") in between.


@torch.no_grad()
def gpt2_mlp(x, mlp):
    """Recompute mlp(x) from its weights. x: (..., 768)."""
    # TODO 1: widen, apply GELU, narrow
    raise NotImplementedError


def swiglu_ffn(x, w_gate, w_up, w_down):
    """SwiGLU feed-forward layer (no biases), as used in LLaMA-style models.

    x: (..., d)   w_gate, w_up: (d, h)   w_down: (h, d)
    out = ( silu(x @ w_gate) * (x @ w_up) ) @ w_down,   silu(z) = z * sigmoid(z)
    Do not call F.silu.
    """
    # TODO 2
    raise NotImplementedError


def ffn_param_count(d, hidden, kind):
    """Weights only, no biases. kind is "mlp" (two matrices) or "swiglu" (three matrices)."""
    # TODO 3
    raise NotImplementedError


def swiglu_hidden_for_same_params(d):
    """The SwiGLU hidden size that matches the parameter count of a standard MLP with hidden size 4d."""
    # TODO 4: solve  3 * d * h = 2 * d * 4d  for h and return an int (round down)
    raise NotImplementedError
