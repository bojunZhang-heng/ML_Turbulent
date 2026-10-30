import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from einops import rearrange
from torch.utils.checkpoint import checkpoint

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


class Physics_Attention_Irregular_Mesh(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape

        ### (1) Slice
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        
        '''
        Compute the slice weights and slice tokens
        '''
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        # Broadcasting avoids materializing a repeated (B,H,G,D) denominator.
        slice_token = slice_token / (slice_norm[..., None] + 1e-5)

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        out_slice_token = F.scaled_dot_product_attention(
            q_slice_token,
            k_slice_token,
            v_slice_token,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale,
        )

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
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


class Transolver_block(nn.Module):
    """Transformer encoder block."""

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
        self.Attn = Physics_Attention_Irregular_Mesh(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                     dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Model(nn.Module):
    def __init__(self,
                 space_dim=3,
                 n_layers=8,
                 n_hidden=256,
                 dropout=0,
                 n_head=8,
                 act='gelu',
                 mlp_ratio=2,
                 fun_dim=0,
                 out_dim=1,
                 slice_num=64,
                 ref=8,
                 unified_pos=False,
                 checkpoint_segment_size=2,
                 checkpoint_num_blocks=None,
                 ):
        super(Model, self).__init__()
        if space_dim != 3:
            raise ValueError(f"Transolver expects space_dim=3, got {space_dim}")
        if fun_dim != 0:
            raise ValueError("Coordinate-only Transolver requires fun_dim=0")
        if n_layers < 1:
            raise ValueError("n_layers must be at least 1")
        if checkpoint_segment_size < 1:
            raise ValueError("checkpoint_segment_size must be at least 1")
        if checkpoint_num_blocks is not None and not 0 <= checkpoint_num_blocks <= n_layers:
            raise ValueError("checkpoint_num_blocks must be between 0 and n_layers")
        self.__name__ = 'UniPDE_3D'
        self.ref = ref
        self.unified_pos = unified_pos
        if self.unified_pos:
            self.preprocess = MLP(fun_dim + self.ref * self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0,
                                  res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.checkpoint_segment_size = checkpoint_segment_size
        self.checkpoint_num_blocks = (
            n_layers if checkpoint_num_blocks is None else checkpoint_num_blocks
        )
        self.checkpoint_preserve_rng_state = dropout > 0.0

        self.blocks = nn.ModuleList([Transolver_block(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      last_layer=False)
                                     for _ in range(n_layers)])
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (n_hidden)) * torch.rand(n_hidden, dtype=torch.float))
        self.Cp_decoder = nn.Linear(n_hidden, out_dim)
        self.integrated_cp_decoder = nn.Linear(1, 1)

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

        gridx = torch.linspace(
            -1.5, 1.5, self.ref, device=my_pos.device, dtype=my_pos.dtype
        )
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat([batchsize, 1, self.ref, self.ref, 1])
        gridy = torch.linspace(
            0, 2, self.ref, device=my_pos.device, dtype=my_pos.dtype
        )
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat([batchsize, self.ref, 1, self.ref, 1])
        gridz = torch.linspace(
            -4, 4, self.ref, device=my_pos.device, dtype=my_pos.dtype
        )
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat([batchsize, self.ref, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).reshape(
            batchsize, self.ref ** 3, 3
        )

        pos = torch.sqrt(
            torch.sum((my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2,
                      dim=-1)). \
            reshape(batchsize, my_pos.shape[1], self.ref * self.ref * self.ref).contiguous()
        return pos

    def _forward_block_range(self, fx, start, end):
        for block_index in range(start, end):
            fx = self.blocks[block_index](fx)
        return fx

    def _forward_with_checkpoint(self, fx, end_block):
        for start in range(0, end_block, self.checkpoint_segment_size):
            end = min(start + self.checkpoint_segment_size, end_block)

            def run_segment(segment_fx, start=start, end=end):
                return self._forward_block_range(segment_fx, start, end)

            fx = checkpoint(
                run_segment,
                fx,
                use_reentrant=False,
                preserve_rng_state=self.checkpoint_preserve_rng_state,
            )
        return fx

    def forward(self, x):
        '''
        x: (B, N, 3) or (B, N, 6)

        Only the first three XYZ channels are used. Six-dimensional
        coordinate-plus-normal inputs remain compatible, but normals are
        intentionally ignored.
        '''
        if not torch.is_tensor(x) or x.ndim != 3:
            raise ValueError("Input must be a tensor shaped (batch, points, channels)")
        if x.shape[-1] < self.space_dim:
            raise ValueError(
                f"Input needs at least {self.space_dim} XYZ channels, got {x.shape[-1]}"
            )

        coords = x[..., :self.space_dim]
        model_input = self.get_grid(coords) if self.unified_pos else coords
        fx = self.preprocess(model_input)
        placeholder = self.placeholder.to(dtype=fx.dtype)
        fx = fx + placeholder[None, None, :]

        checkpoint_end = self.checkpoint_num_blocks if self.training else 0
        if checkpoint_end:
            fx = self._forward_with_checkpoint(fx, checkpoint_end)
        fx = self._forward_block_range(fx, checkpoint_end, len(self.blocks))
        
        # Make the prediction for surface pressure
        Cp_pred = self.Cp_decoder(fx)

        return Cp_pred
