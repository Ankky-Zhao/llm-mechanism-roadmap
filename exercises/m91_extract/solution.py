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
    out = model(ids[None], output_attentions=True)
    return torch.stack([a[0] for a in out.attentions])


def attention_entropy(att):
    """Entropy in nats of every attention row, averaged over the positions. att (L, H, T, T) -> (L, H)."""
    ent = -(att * torch.log(att.clamp_min(1e-12))).sum(-1)
    return ent.mean(-1)


def plot_head(att, tokens, layer, head, path):
    """Save a heatmap of one head. Rows are the reading positions, columns the positions read."""
    a = att[layer, head].numpy()
    fig, ax = plt.subplots(figsize=(0.45 * len(tokens) + 1.5, 0.45 * len(tokens) + 1.5))
    im = ax.imshow(a, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens)
    ax.set_xlabel("position read (key)")
    ax.set_ylabel("reading position (query)")
    ax.set_title(f"layer {layer} head {head}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
