import torch


def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)       # biased: divide by d, not d - 1
    return (x - mean) / torch.sqrt(var + eps) * weight + bias


def rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * weight
