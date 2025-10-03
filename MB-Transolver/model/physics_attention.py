import torch.nn as nn
import torch
import logging
from einops import rearrange, repeat


class PhysicsAttention(nn.Module):
    def __init__(self, args):
        super().__init__(args)

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, args.n_hidden, bias=False),
            nn.Dropout(args.dropout)
        )
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim_head = args.dim_head
        self.inner_dim = args.dim_head * args.n_heads
        self.scale = args.dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(args.dropout)
        self.temperature = nn.Parameter(torch.ones([1, args.n_heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(args.n_hidden, self.inner_dim)
        self.in_project_fx = nn.Linear(args.n_hidden, self.inner_dim)
        self.in_project_slice = nn.Linear(args.dim_head, args.slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(args.dim_head, args.dim_head, bias=False)
        self.to_k = nn.Linear(args.dim_head, args.dim_head, bias=False)
        self.to_v = nn.Linear(args.dim_head, args.dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, args.n_hidden),
            nn.Dropout(args.dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape

        ### (1) Slice
        fx_mid = self.in_project_fx(x).reshape(B, N, self.n_heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C (batch_size, num_heads, num_points, channel)
        x_mid = self.in_project_x(x).reshape(B, N, self.n_heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C (batch_size, num_heads, num_points, channel)
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G (batch_size, num_heads, num_points, slice_num)
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D (batch_size, num_heads, slice_num, dim_head)

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')     # 4*64 = 256
        return self.to_out(out_x)



