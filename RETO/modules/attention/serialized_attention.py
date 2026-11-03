import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from einops import rearrange

class Serialized_Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            patch_size = 20,
            shift = 2,
            dropout = 0.1):
        super(Serialized_Attention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.patch_size = patch_size
        self.shift = shift
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        index = torch.tensor([i for i in range(0, patch_size*shift, shift)], dtype=torch.int64)[None, ...]
        self.group_index = torch.cat([index+i for i in range(shift)], dim=0)

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # padding
        pad_size = int((self.patch_size*self.shift) - N % (self.patch_size*self.shift))
        x_pad = torch.cat([x, torch.zeros(B, pad_size, C).to(x.device)], dim=1)

        # index_generation
        index = self.group_index
        index_list = [self.group_index]
        while index[-1,-1].item() < N:
            index = index + (self.patch_size*self.shift)
            index_list.append(index)
        patch_index = torch.cat(index_list, dim=0)

        # pad2patch
        x_patch = x_pad[:, patch_index, :] # (B, patch_num, patch_size, C)

        # patch attention
        x_patch = rearrange(x_patch, 'b n s c -> (b n) s c')
        B_p, S_p, C = x_patch.shape

        qkv = self.qkv_proj(x_patch)
        qkv = qkv.reshape(B_p, S_p, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q_token, k_token, v_token = qkv[0], qkv[1], qkv[2]

        dots = torch.matmul(q_token, k_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        attn_token = torch.matmul(attn, v_token)

        attn_token = attn_token.transpose(1, 2).reshape(B_p, S_p, self.dim)
        out_token = attn_token

        out_token = rearrange(out_token, '(b n) s c -> b n s c', b=B)

        # patch2pad
        x_pad[:, patch_index, :] = out_token

        return self.out_proj(x_pad[:, :N, :])


