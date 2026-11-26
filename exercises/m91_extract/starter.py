import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_gpt2():
    """GPT-2 small. attn_implementation="eager" makes the model return its attention weights."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").eval()
    return model, tok


@torch.no_grad()
def get_attention(model, ids):
    """ids: LongTensor (T,). Returns the attention weights of all heads, shape (n_layer, n_head, T, T).

    Entry [l, h, i, j] is the weight with which head h of layer l, at position i, reads position j.
    """
    # TODO 1: call the model with output_attentions=True. out.attentions is a tuple with one
    #         tensor of shape (1, n_head, T, T) per layer. Stack them into one tensor.
    raise NotImplementedError


def attention_entropy(att):
    """Entropy in nats of every attention row, averaged over the positions. att (L, H, T, T) -> (L, H)."""
    # TODO 2: H = -sum_j a_ij log a_ij. Clamp before the log, masked entries are exactly 0.
    raise NotImplementedError


def plot_head(att, tokens, layer, head, path):
    """Save a heatmap of one head. Rows are the reading positions, columns the positions read."""
    # TODO 3: plt.subplots, ax.imshow(att[layer, head], cmap="Blues", vmin=0, vmax=1),
    #         label both axes with the tokens, fig.savefig(path), plt.close(fig)
    raise NotImplementedError
