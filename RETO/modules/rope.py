import einops
import torch


def rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """RoPE via polar coordinate rotations."""
    # adapted from https://github.com/meta-llama/llama3/blob/main/llama/model.py#L65
    assert x.ndim == 4, "x.shape should be (batch_size, num_heads, seqlen, head_dim)"
    assert freqs.ndim == 4, "freqs.shape should be (batch_size, num_heads, seqlen, head_dim)"
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_ = torch.view_as_complex(freqs.float().reshape(*freqs.shape[:-1], -1, 2).contiguous())
    x_out = torch.view_as_real(x_ * freqs_).flatten(start_dim=3)

    return x_out.type_as(x)




