import numpy as np
from timm.layers.weight_init import trunc_normal_
from models.Embedding import timestep_embedding

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class Serialized_Attention(nn.Module):
    def __init__(self, patch_size, shift, dim, num_heads, dropout=0.1):
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

class LinearNO(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.,key_ratio=4):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.tempreature_q = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.tempreature_k = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.to_q = nn.Linear(dim_head, key_ratio*dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, key_ratio*dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.GELU(),
                                    nn.Linear(dim, dim), nn.Dropout(dropout))
    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x_mid = self.in_project_x(x)
        x_mid = rearrange(x_mid, 'b n (h d) -> b h n d',h=self.heads, d=self.dim_head)
        q=self.to_q(x_mid)
        k=self.to_k(x_mid)
        v=self.to_v(x_mid)

        q=F.softmax(q/torch.clamp(self.tempreature_q,max=2,min=0.1),dim=-1)
        k=F.softmax(k/torch.clamp(self.tempreature_k,max=2,min=0.1),dim=-2)

        kv = torch.einsum("bhnd,bhnc->bhdc", k, v)
        qkv = torch.einsum("bhnd,bhdc->bhnc", q, kv)
        qkv=rearrange(qkv, 'b h n d -> b n (h d)')
        return self.to_out(qkv)

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x

# Linear Attention Neural Operator
class LinearNO_block(nn.Module):

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act='gelu',
        mlp_ratio=4,
        key_ratio=4,
        last_layer=False,
        out_dim=1,
        H=85,
        W=85,
        isregular=False,
        patch_size=20,
        shift=1
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)

        self.global_attention = LinearNO(hidden_dim,
                                         heads=num_heads,
                                         dim_head=hidden_dim // num_heads,
                                         dropout=dropout,
                                         key_ratio=key_ratio)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.ln_3 = nn.LayerNorm(hidden_dim)


        self.local_ln_1 = nn.LayerNorm(hidden_dim)
        self.local_ln_2 = nn.LayerNorm(hidden_dim)
        self.local_gate = nn.Parameter(torch.tensor([0.0]))
        self.local_attention = Serialized_Attention(patch_size, shift, hidden_dim, num_heads, dropout=0.1)

        self.mlp = MLP(hidden_dim,
                       hidden_dim * mlp_ratio,
                       hidden_dim,
                       n_layers=0,
                       res=False,
                       act=act)
        if self.last_layer:
            self.ln_4 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)
    def forward(self, fx, order, inverse):
        x0 = fx 
        fxn = self.ln_1(fx)

        # global_attention
        fx1 = self.global_attention(fxn)
        
        # local_attention
        fx2 = torch.zeros_like(fx1)
        for i in range(fx2.shape[0]):
            fx2[i] = (fxn - fx1)[i, order[i], :] # serialized
        fx2 = self.local_ln_2(self.local_attention(self.local_ln_1(fx2)))
        for i in range(fx2.shape[0]):
            fx2[i] = fx2[i, inverse[i], :]  # deserialized
        
        fx = self.mlp(self.ln_3(x0 + fx1 + self.local_gate * fx2)) + x0
        
        if self.last_layer:
            return self.mlp2(self.ln_4(fx))
        else:
            return fx


class LinearAttentionNeuralOperator(nn.Module):

    def __init__(
        self,
        space_dim=1,
        n_layers=5,
        n_hidden=256,
        dropout=0.0,
        n_head=8,
        Time_Input=False,
        act='gelu',
        mlp_ratio=1,
        fun_dim=1,
        out_dim=1,
        key_ratio=4,
        ref=8,
        unified_pos=False,
        H=85,
        W=85,
        isregular=False,
        patch_size=20,
        shift=1,
    ):
        super(LinearAttentionNeuralOperator, self).__init__()
        self.H = H
        self.W = W
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.pos = self.get_grid()
            self.preprocess = MLP(fun_dim + self.ref * self.ref,
                                  n_hidden * 2,
                                  n_hidden,
                                  n_layers=0,
                                  res=False,
                                  act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim,
                                  n_hidden * 2,
                                  n_hidden,
                                  n_layers=0,
                                  res=False,
                                  act=act)

        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden),
                                         nn.SiLU(),
                                         nn.Linear(n_hidden, n_hidden))

        self.blocks = nn.ModuleList([
            LinearNO_block(num_heads=n_head,
                        hidden_dim=n_hidden,
                        dropout=dropout,
                        act=act,
                        mlp_ratio=mlp_ratio,
                        out_dim=out_dim,
                        key_ratio=key_ratio,
                        H=H,
                        W=W,
                        isregular=isregular,
                        patch_size=patch_size,
                        shift=shift,
                        last_layer=(_ == n_layers - 1))
            for _ in range(n_layers)
        ])
        self.initialize_weights()
        self.placeholder = nn.Parameter(
            (1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_grid(self, batchsize=1):
        size_x, size_y = self.H, self.W
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1,
                              1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y,
                              1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).cuda()  # B H W 2

        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1,
                              1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref,
                              1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).cuda()  # B H W 8 8 2

        pos = torch.sqrt(
            torch.sum((grid[:, :, :, None, None, :] -
                       grid_ref[:, None, None, :, :, :])**2,
                      dim=-1)).reshape(batchsize, size_x, size_y,
                                       self.ref * self.ref).contiguous()
        return pos

    def forward(self,data, order, inverse):
        x, fx, T = data[...,:self.space_dim], None, None
        if self.unified_pos:
            x = self.pos.repeat(x.shape[0], 1, 1,
                                1).reshape(x.shape[0], self.H * self.W,
                                           self.ref * self.ref)
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        if T is not None:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(
                1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for i, block in enumerate(self.blocks):
            serialization_type = int(i % order.shape[1])
            fx = block(fx, order[:, serialization_type, :], inverse[:, serialization_type, :])

        return fx[0]
