import torch
import torch.nn as nn
from einops import rearrange

from models.transolver_model import MLP, Physics_Attention_Irregular_Mesh
from SATO_dataset.Dataset import pos_to_order_inverse_index


class SerializedAttention(nn.Module):
    def __init__(self, patch_size, shift, dim, num_heads, dropout=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if patch_size <= 0 or shift <= 0:
            raise ValueError("patch_size and shift must be positive")

        self.patch_size = int(patch_size)
        self.shift = int(shift)
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        base_index = (
            torch.arange(self.shift)[:, None]
            + torch.arange(self.patch_size)[None, :] * self.shift
        )
        self.register_buffer("base_index", base_index, persistent=False)
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, num_points, channels = x.shape
        block_size = self.patch_size * self.shift
        pad_size = (-num_points) % block_size
        if pad_size:
            x_pad = torch.cat([x, x.new_zeros(batch_size, pad_size, channels)], dim=1)
        else:
            x_pad = x

        num_blocks = x_pad.shape[1] // block_size
        block_offsets = (
            torch.arange(num_blocks, device=x.device)[:, None, None] * block_size
        )
        patch_index = (block_offsets + self.base_index[None]).reshape(
            num_blocks * self.shift, self.patch_size
        )

        x_patch = x_pad[:, patch_index, :]
        x_patch = rearrange(x_patch, "b p s c -> (b p) s c")
        patch_batch, patch_length, _ = x_patch.shape

        qkv = self.qkv_proj(x_patch).reshape(
            patch_batch,
            patch_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = torch.softmax(
            torch.matmul(query, key.transpose(-1, -2)) * self.scale,
            dim=-1,
        )
        attention = self.dropout(attention)
        output = torch.matmul(attention, value)
        output = output.transpose(1, 2).reshape(
            patch_batch, patch_length, self.dim
        )
        output = rearrange(
            output,
            "(b p) s c -> b p s c",
            b=batch_size,
            p=num_blocks * self.shift,
        )

        restored = torch.empty_like(x_pad)
        restored[:, patch_index, :] = output.to(dtype=restored.dtype)
        return self.out_proj(restored[:, :num_points, :])


class SATOBlock(nn.Module):
    def __init__(
        self,
        num_heads,
        hidden_dim,
        dropout,
        act="gelu",
        mlp_ratio=2,
        slice_num=64,
        patch_size=20,
        shift=2,
    ):
        super().__init__()
        self.global_norm = nn.LayerNorm(hidden_dim)
        self.global_attention = Physics_Attention_Irregular_Mesh(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
        )
        self.local_input_norm = nn.LayerNorm(hidden_dim)
        self.local_output_norm = nn.LayerNorm(hidden_dim)
        self.local_attention = SerializedAttention(
            patch_size,
            shift,
            hidden_dim,
            num_heads,
            dropout=0.1,
        )
        self.local_gate = nn.Parameter(torch.zeros(1))
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )

    @staticmethod
    def _reorder(features, index):
        gather_index = index.unsqueeze(-1).expand(-1, -1, features.shape[-1])
        return torch.gather(features, dim=1, index=gather_index)

    def forward(self, features, order, inverse):
        residual = features
        normalized = self.global_norm(features)
        global_features = self.global_attention(normalized)

        local_features = self._reorder(normalized - global_features, order)
        local_features = self.local_attention(self.local_input_norm(local_features))
        local_features = self.local_output_norm(local_features)
        local_features = self._reorder(local_features, inverse)

        combined = residual + global_features + self.local_gate * local_features
        return residual + self.mlp(self.mlp_norm(combined))


class Model(nn.Module):
    """SATO for seven-channel MLCFD input and four-channel CFD prediction."""

    def __init__(
        self,
        space_dim=7,
        n_layers=8,
        n_hidden=256,
        dropout=0.0,
        n_head=8,
        act="gelu",
        mlp_ratio=2,
        fun_dim=0,
        slice_num=64,
        patch_size=20,
        shift=2,
        n_iter=1,
        out_dim=4,
    ):
        super().__init__()
        if n_hidden % n_head != 0:
            raise ValueError("n_hidden must be divisible by n_head")
        if n_layers <= 0 or n_iter <= 0:
            raise ValueError("n_layers and n_iter must be positive")
        if fun_dim != 0:
            raise ValueError("ShapeNet SATO expects every input channel in space_dim")

        self.space_dim = int(space_dim)
        self.out_dim = int(out_dim)
        self.n_layers = int(n_layers)
        self.n_iter = int(n_iter)
        self.preprocess = MLP(
            space_dim,
            n_hidden * 2,
            n_hidden,
            n_layers=0,
            res=False,
            act=act,
        )
        self.placeholder = nn.Parameter(torch.rand(n_hidden) / n_hidden)
        self.blocks = nn.ModuleList(
            [
                SATOBlock(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    slice_num=slice_num,
                    patch_size=patch_size,
                    shift=shift,
                )
                for _ in range(n_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(n_hidden)
        self.output_decoder = nn.Linear(n_hidden, out_dim)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)

    def forward(self, data, order=None, inverse=None):
        if data.ndim != 3 or data.shape[-1] != self.space_dim:
            raise ValueError(
                f"Expected input shape (B, N, {self.space_dim}), got {tuple(data.shape)}"
            )
        if order is None or inverse is None:
            order, inverse = pos_to_order_inverse_index(
                data[..., :3], data.device, tensor=True
            )
        expected_index_shape = (data.shape[0], 4, data.shape[1])
        if tuple(order.shape) != expected_index_shape:
            raise ValueError(
                f"Expected serialization indices {expected_index_shape}, got {tuple(order.shape)}"
            )

        features = self.preprocess(data) + self.placeholder[None, None, :]
        for _ in range(self.n_iter):
            for layer_index, block in enumerate(self.blocks):
                serialization_type = layer_index % order.shape[1]
                features = block(
                    features,
                    order[:, serialization_type, :],
                    inverse[:, serialization_type, :],
                )
        return self.output_decoder(self.output_norm(features))


LinearAttentionNeuralOperator = Model
