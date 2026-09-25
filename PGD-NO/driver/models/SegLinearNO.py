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

#class Physics_Attention_seg(nn.Module):
class SegLinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        key_ratio=4,
    ):
        super().__init__()

        inner_dim = heads * dim_head
        key_dim = key_ratio * dim_head

        self.dim_head = dim_head
        self.heads = heads
        self.key_dim = key_dim

        self.in_project_x = nn.Linear(dim, inner_dim)

        # 节点 Query
        self.to_q = nn.Linear(dim_head, key_dim, bias=False)

        # 几何 token Key
        self.to_k = nn.Linear(dim_head, key_dim, bias=False)

        # 几何 token Value
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.temperature_q = nn.Parameter(
            torch.ones(1, heads, 1, 1) * 0.5
        )

        self.temperature_k = nn.Parameter(
            torch.ones(1, heads, 1, 1) * 0.5
        )

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, seg_matrix):
        """
        x:
            [B, N, C]

        seg_matrix:
            [B, S, N]

        return:
            [B, N, C]
        """

        B, N, _ = x.shape
        _, S, N_seg = seg_matrix.shape

        if N != N_seg:
            raise ValueError(
                f"x and seg_matrix node dimensions do not match: "
                f"x has N={N}, seg_matrix has N={N_seg}"
            )

        # -------------------------------------------------
        # 1. 节点特征投影
        # -------------------------------------------------
        x_mid = self.in_project_x(x)

        x_mid = rearrange(
            x_mid,
            "b n (h d) -> b h n d",
            h=self.heads,
            d=self.dim_head,
        )
        # [B, H, N, D]

        # -------------------------------------------------
        # 2. 使用 seg_matrix 聚合几何 token
        # -------------------------------------------------
        geometry_tokens = torch.einsum(
            "bsn,bhnd->bhsd",
            seg_matrix,
            x_mid,
        )
        # [B, H, S, D]

        # -------------------------------------------------
        # 3. 节点生成 Query
        # -------------------------------------------------
        q = self.to_q(x_mid)
        # [B, H, N, key_dim]

        # -------------------------------------------------
        # 4. 区域 token 生成 Key/Value
        # -------------------------------------------------
        k = self.to_k(geometry_tokens)
        # [B, H, S, key_dim]

        v = self.to_v(geometry_tokens)
        # [B, H, S, D]

        # -------------------------------------------------
        # 5. Linear Attention 归一化
        # -------------------------------------------------
        temperature_q = torch.clamp(
            self.temperature_q,
            min=0.1,
            max=2.0,
        )

        temperature_k = torch.clamp(
            self.temperature_k,
            min=0.1,
            max=2.0,
        )

        # Query 在 key/channel 维度归一化
        q = F.softmax(
            q / temperature_q,
            dim=-1,
        )
        # [B, H, N, key_dim]

        # Key 在区域维度归一化
        #
        # 与原始 LinearNO 不同：
        # 原始 LinearNO 的 K 长度为 N
        # 现在几何 token 的长度为 S
        k = F.softmax(
            k / temperature_k,
            dim=-2,
        )
        # [B, H, S, key_dim]

        # -------------------------------------------------
        # 6. 先聚合区域 token 的 K/V
        # -------------------------------------------------
        kv = torch.einsum(
            "bhsk,bhsd->bhkd",
            k,
            v,
        )
        # [B, H, key_dim, D]

        # -------------------------------------------------
        # 7. 每个节点 Query 区域信息
        # -------------------------------------------------
        qkv = torch.einsum(
            "bhnk,bhkd->bhnd",
            q,
            kv,
        )
        # [B, H, N, D]

        # -------------------------------------------------
        # 8. 合并多头
        # -------------------------------------------------
        qkv = rearrange(
            qkv,
            "b h n d -> b n (h d)",
        )
        # [B, N, inner_dim]

        return self.to_out(qkv)
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
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        #self.Attn = Physics_Attention_seg(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
        #                                 dropout=dropout, slice_num=slice_num)
        self.Attn = SegLinearAttention(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                          dropout=dropout)
        

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, seg_matrix, return_attention=False):
        if return_attention:
            # When extracting attention, don't use checkpointing
            attn_output = self.Attn(self.ln_1(fx), seg_matrix, return_attention=True)
            out_x, attn_coord_to_token = attn_output
            fx = out_x + fx
        else:
            if self.training:
                fx = checkpoint(self.Attn, self.ln_1(fx), seg_matrix, use_reentrant=True) + fx
            else:
                fx = self.Attn(self.ln_1(fx), seg_matrix) + fx
        
        if self.training and not return_attention:
            fx = checkpoint(self.mlp, self.ln_2(fx), use_reentrant=True) + fx
        else:
            fx = self.mlp(self.ln_2(fx)) + fx
        
        if self.last_layer:
            result = self.mlp2(self.ln_3(fx))
            if return_attention:
                return result, attn_coord_to_token
            else:
                return result
        else:
            if return_attention:
                return fx, attn_coord_to_token
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

        # 给所有节点添加同一个可训练向量，非必需参数，为兼容“无输入物理场设计”
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

    def forward(self, data, return_attention=False):
        x, seg_matrix = data

        fx, T = None, None
        if fx is not None:
            fx = torch.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]    # (B, N, C)
        
        all_attentions = []
        
        # geometry information processing
        for i, block in enumerate(self.blocks):
            if return_attention:
                fx, attn_coord_to_token = block(fx, seg_matrix, return_attention=True)
                all_attentions.append(attn_coord_to_token)
            else:
                fx = block(fx, seg_matrix)    # (B, N, F)
        
        fx = self.Cp_decoder(fx)

        if return_attention:
            return fx, all_attentions
        else:
            return fx
    
    def extract_attention_scores(self, data):
        """
        Extract attention scores between coordinates (query nodes) and slice tokens across all layers.
        
        Args:
            data: Tuple of (x, seg_matrix) where x is (B, N, F) and seg_matrix is (B, S, N)
            
        Returns:
            dict: Dictionary containing:
                - 'attention_scores': List of (B, H, N, S) numpy arrays, one per layer,
                  representing attention from coordinates (query nodes) to slice tokens
                  where B=batch, H=heads, N=num_query_nodes, S=num_tokens
        """
        self.eval()
        with torch.no_grad():
            _, all_attentions = self.forward(data, return_attention=True)
            
            # Convert to numpy arrays
            attention_scores = [attn.cpu().numpy() for attn in all_attentions]
            
            return {
                'attention_scores': attention_scores,  # List of (B, H, N, S) arrays
            }
