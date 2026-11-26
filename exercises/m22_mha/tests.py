import torch
import torch.nn as nn

torch.manual_seed(0)
B, T, d = 2, 6, 64
x = torch.randn(B, T, d)

for h in (1, 4, 8):
    ref = nn.MultiheadAttention(d, h, batch_first=True)
    mine = MultiHeadAttention(d, h)
    with torch.no_grad():                              # same weights in both modules
        mine.qkv.weight.copy_(ref.in_proj_weight); mine.qkv.bias.copy_(ref.in_proj_bias)
        mine.proj.weight.copy_(ref.out_proj.weight); mine.proj.bias.copy_(ref.out_proj.bias)

    out, w = mine(x)
    ref_out, ref_w = ref(x, x, x, need_weights=True, average_attn_weights=False)
    assert out.shape == (B, T, d) and w.shape == (B, h, T, T), f"n_head={h}: wrong shapes"
    assert torch.allclose(out, ref_out, atol=1e-5), f"n_head={h}: output differs from nn.MultiheadAttention"
    assert torch.allclose(w, ref_w, atol=1e-5), f"n_head={h}: attention weights differ"

    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))            # True = may attend
    out_c, w_c = mine(x, mask=causal)
    # note: nn.MultiheadAttention uses the opposite convention, True = NOT allowed
    ref_c, _ = ref(x, x, x, attn_mask=~causal, need_weights=False)
    assert torch.allclose(out_c, ref_c, atol=1e-5), f"n_head={h}: masked output differs"
    assert (w_c[..., ~causal] == 0).all(), "no weight may fall on future positions"

    n_params = sum(p.numel() for p in ref.parameters())
    assert mha_param_count(d, h) == n_params == sum(p.numel() for p in mine.parameters())

assert mha_param_count(768, 12) == 2_362_368, "one GPT-2 small attention layer has 2,362,368 parameters"
print("m22_mha: all tests passed")
