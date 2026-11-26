import torch
import torch.nn as nn

torch.manual_seed(0)
d_in, d_out, r = 64, 48, 4
base = nn.Linear(d_in, d_out)
w0 = base.weight.detach().clone()
layer = LoRALinear(base, r=r, alpha=8)
x = torch.randn(10, d_in)

# 1. at initialisation nothing changes
assert torch.allclose(layer(x), base(x)), "B starts at zero, so the output must equal the frozen layer's"
assert layer.delta_weight().abs().max() == 0

# 2. only A and B are trainable
trainable = {n for n, p in layer.named_parameters() if p.requires_grad}
assert trainable == {"A", "B"}, f"trainable parameters: {trainable}"
n_train = sum(p.numel() for p in layer.parameters() if p.requires_grad)
assert n_train == lora_param_count(d_in, d_out, r) == r * (d_in + d_out)
assert lora_param_count(768, 2304, 8) == 24576 and 768 * 2304 == 1769472, "72 times fewer than the full matrix of GPT-2's c_attn"

# 3. the layer can learn a low-rank change of the weights
target_delta = torch.randn(d_out, 2) @ torch.randn(2, d_in) * 0.1          # a rank-2 change
y = x.new_empty(0)
X = torch.randn(512, d_in)
Y = X @ (w0 + target_delta).T + base.bias.detach()
opt = torch.optim.Adam([layer.A, layer.B], lr=1e-2)
for _ in range(800):
    loss = ((layer(X) - Y) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
assert loss.item() < 1e-4, f"final loss {loss.item():.2e}"
assert torch.equal(base.weight, w0), "the frozen weight must not move"
assert torch.linalg.matrix_rank(layer.delta_weight(), atol=1e-4) <= r, "the update has rank at most r"
assert torch.allclose(layer.delta_weight(), target_delta, atol=2e-2), "the learned update should match the hidden one"

# 4. merging gives an ordinary layer with identical outputs, so inference costs nothing extra
merged = layer.merge()
assert isinstance(merged, nn.Linear) and not hasattr(merged, "A")
assert torch.allclose(merged(x), layer(x), atol=1e-5)
print(f"m45_lora_linear: all tests passed ({n_train} trainable of {d_in * d_out + d_out} parameters)")
