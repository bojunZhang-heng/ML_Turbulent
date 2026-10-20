import torch
import numpy as np
import torch.nn as nn
from timm.layers import trunc_normal_
from einops import rearrange
import torch.distributed.nn as dist_nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}

def matmul_single(fx_mid, slice_weights):
    return fx_mid.T @ slice_weights

def gumbel_softmax(logits, tau=1, hard=False):
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)

    y = logits + gumbel_noise
    y = y / tau
    
    y = F.softmax(y, dim=-1)
    
    if hard:
        _, y_hard = y.max(dim=-1)
        y_one_hot = torch.zeros_like(y).scatter_(-1, y_hard.unsqueeze(-1), 1.0)
        y = (y_one_hot - y).detach() + y
    return y

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
    
class Physics_Attention_seg(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU()
        )

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.token_linear = nn.Linear(dim_head, dim_head, bias=False)

        self.self_to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.self_to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.self_to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.cross_to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.cross_to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.cross_to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, seg_matrix):
        '''
        x: (B, N, C)
        seg_matrix: (B, S, N), where S is number of physics tokens
        '''
        # B N C
        B, N, C = x.shape
        
        # compute the slice tokens
        query_x = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_token = torch.einsum("bsn,bhnc->bhsc", seg_matrix, x_mid)  # B H S C
        out_slice_token = self.token_linear(slice_token)

        # self attention on the tokens
        q_token = self.self_to_q(out_slice_token)
        k_token = self.self_to_k(out_slice_token)
        v_token = self.self_to_v(out_slice_token)
        out_slice_token = F.scaled_dot_product_attention(q_token, k_token, v_token)

        # cross attention on the slice tokens and the data coordinates
        q_x = self.cross_to_q(query_x)
        k_slice_token = self.cross_to_k(out_slice_token)
        v_slice_token = self.cross_to_v(out_slice_token)
        out_x = F.scaled_dot_product_attention(q_x, k_slice_token, v_slice_token)

        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)

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


class Transolver_plus_block(nn.Module):
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
            patch_size=20,
            shift=1
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.global_attention = Physics_Attention_seg(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                         dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        
        self.local_ln_1 = nn.LayerNorm(hidden_dim)
        self.local_attention = Serialized_Attention(patch_size, shift, hidden_dim, num_heads, dropout=0.1)
        self.local_ln_2 = nn.LayerNorm(hidden_dim)
        self.local_gate = nn.Parameter(torch.tensor([0.0]))

        self.ln_3 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)

        if self.last_layer:
            self.ln_4 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, seg_matrix, order, inverse):
        x0 = fx 
        fxn = self.ln_1(fx)

        # global_attention
        fx1 = self.global_attention(fxn, seg_matrix)
    
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


class Model(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0,
                 n_head=8,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False
                 ):
        super(Model, self).__init__()
        self.__name__ = 'UniPDE_3D'
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.preprocess = MLP(fun_dim + self.ref * self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0,
                                  res=False, act=act)
        else:
            self.preprocess = MLP(space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.embedding = nn.Linear(3, n_hidden)
        self.blocks = nn.ModuleList([Transolver_plus_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      last_layer=False)
                                     for _ in range(n_layers)])
        self.initialize_weights()
        self.Cp_decoder = nn.Linear(n_hidden, out_dim)
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))

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

    def get_grid(self, my_pos):
        # my_pos 1 N 3
        batchsize = my_pos.shape[0]

        gridx = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat([batchsize, 1, self.ref, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 2, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat([batchsize, self.ref, 1, self.ref, 1])
        gridz = torch.tensor(np.linspace(-4, 4, self.ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat([batchsize, self.ref, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).cuda().reshape(batchsize, self.ref ** 3, 3)  # B 4 4 4 3

        pos = torch.sqrt(
            torch.sum((my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2,
                      dim=-1)). \
            reshape(batchsize, my_pos.shape[1], self.ref * self.ref * self.ref).contiguous()
        return pos

    def forward(self, x, seg_matrix, order, inverse):

        fx, T = None, None
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]    # (B, N, C)
        
        # geometry information processing
        for i, block in enumerate(self.blocks):
            serialization_type = int(i % order.shape[1])
            fx = block(fx, seg_matrix, order[:, serialization_type, :], inverse[:, serialization_type, :])
        
        fx = self.Cp_decoder(fx)

        return fx
