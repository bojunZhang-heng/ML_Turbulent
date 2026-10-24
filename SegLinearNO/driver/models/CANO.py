from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers.weight_init import trunc_normal_
from torch.utils.checkpoint import checkpoint


ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    def __init__(
        self, n_input, n_hidden, n_output, n_layers=1, act="gelu", res=True
    ):
        super().__init__()
        if act not in ACTIVATION:
            raise NotImplementedError(f"Unsupported activation: {act}")

        activation = ACTIVATION[act]
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), activation())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList(
            nn.Sequential(nn.Linear(n_hidden, n_hidden), activation())
            for _ in range(n_layers)
        )

    def forward(self, x):
        x = self.linear_pre(x)
        for layer in self.linears:
            x = layer(x) + x if self.res else layer(x)
        return self.linear_post(x)


class ClusterAttention(nn.Module):
    """CANO attention that clusters queries while preserving full K and V."""

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        key_ratio=1,
        cluster_num=64,
        **_,
    ):
        super().__init__()
        if cluster_num < 1:
            raise ValueError("cluster_num must be at least 1")

        inner_dim = dim_head * heads
        key_dim = key_ratio * dim_head
        self.dim_head = dim_head
        self.key_dim = key_dim
        self.heads = heads
        self.cluster_num = cluster_num
        self.scale = key_dim**-0.5
        self.attention_dropout = dropout

        # Q, K and V are projected independently from the complete hidden
        # feature. Each output head occupies its own channel slice, so neither
        # the three projections nor the per-head projection weights are shared.
        self.to_q = nn.Linear(dim, heads * key_dim, bias=False)
        self.to_k = nn.Linear(dim, heads * key_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.cluster_projection = nn.Parameter(
            torch.empty(heads, key_dim, cluster_num)
        )
        trunc_normal_(self.cluster_projection, std=0.02)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def _project_qkv(self, x):
        q = rearrange(
            self.to_q(x),
            "b n (h d) -> b h n d",
            h=self.heads,
            d=self.key_dim,
        )
        k = rearrange(
            self.to_k(x),
            "b n (h d) -> b h n d",
            h=self.heads,
            d=self.key_dim,
        )
        v = rearrange(
            self.to_v(x),
            "b n (h d) -> b h n d",
            h=self.heads,
            d=self.dim_head,
        )
        return q, k, v

    def _cluster_assignment(self, q):
        cluster_logits = torch.einsum(
            "bhnd,hdk->bhnk", q, self.cluster_projection
        )
        return F.softmax(cluster_logits, dim=-1)

    def _cluster_queries(self, q):
        cluster_assignment = self._cluster_assignment(q)
        clustered_q = torch.einsum(
            "bhnk,bhnd->bhkd", cluster_assignment, q
        )
        return cluster_assignment, clustered_q

    def _cross_attention(self, clustered_q, k, v):
        return F.scaled_dot_product_attention(
            clustered_q,
            k,
            v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scale,
        )

    def _broadcast_clusters(self, cluster_assignment, cluster_features):
        return torch.einsum(
            "bhnk,bhkd->bhnd", cluster_assignment, cluster_features
        )

    def forward(self, x):
        q, k, v = self._project_qkv(x)
        cluster_assignment, clustered_q = self._cluster_queries(q)
        cluster_features = self._cross_attention(clustered_q, k, v)
        output = self._broadcast_clusters(cluster_assignment, cluster_features)
        output = rearrange(output, "b h n d -> b n (h d)")
        return self.to_out(output)


class CANOBlock(nn.Module):
    def __init__(
        self,
        num_heads,
        hidden_dim,
        dropout,
        act="gelu",
        mlp_ratio=4,
        key_ratio=1,
        cluster_num=64,
        last_layer=False,
        out_dim=1,
        attention_cls=ClusterAttention,
        attention_kwargs=None,
        **_,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = attention_cls(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            key_ratio=key_ratio,
            cluster_num=cluster_num,
            **(attention_kwargs or {}),
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim,
            int(hidden_dim * mlp_ratio),
            hidden_dim,
            n_layers=0,
            res=False,
            act=act,
        )
        if last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(
        self, fx: torch.Tensor, attn_kwargs: dict[str, Any] | None = None
    ) -> torch.Tensor:
        fx = self.attn(self.ln_1(fx), **(attn_kwargs or {})) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class ClusterAttentionNeuralOperator(nn.Module):
    """Coordinate-only Cluster Attention Neural Operator."""

    def __init__(
        self,
        space_dim=3,
        n_layers=5,
        n_hidden=256,
        dropout=0.0,
        n_head=8,
        Time_Input=False,
        act="gelu",
        mlp_ratio=1,
        fun_dim=0,
        out_dim=1,
        key_ratio=1,
        ref=8,
        unified_pos=False,
        H=85,
        W=85,
        isregular=False,
        cluster_num=64,
        checkpoint_segment_size=2,
        _attention_cls=ClusterAttention,
        _attention_kwargs=None,
        **_,
    ):
        super().__init__()
        if space_dim != 3:
            raise ValueError(f"CANO expects space_dim=3, got {space_dim}")
        if fun_dim != 0:
            raise ValueError("CANO is coordinate-only, so fun_dim must be 0")
        if unified_pos or Time_Input:
            raise ValueError("unified_pos and Time_Input are not supported")
        if checkpoint_segment_size < 1:
            raise ValueError("checkpoint_segment_size must be at least 1")

        self.space_dim = space_dim
        self.n_hidden = n_hidden
        self.cluster_num = cluster_num
        self.checkpoint_segment_size = checkpoint_segment_size
        self.preprocess = MLP(
            space_dim,
            n_hidden * 2,
            n_hidden,
            n_layers=0,
            res=False,
            act=act,
        )
        self.blocks = nn.ModuleList(
            CANOBlock(
                num_heads=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                key_ratio=key_ratio,
                cluster_num=cluster_num,
                out_dim=out_dim,
                attention_cls=_attention_cls,
                attention_kwargs=_attention_kwargs,
                last_layer=layer_index == n_layers - 1,
                H=H,
                W=W,
                isregular=isregular,
            )
            for layer_index in range(n_layers)
        )
        self.initialize_weights()
        self.placeholder = nn.Parameter(torch.rand(n_hidden) / n_hidden)

    def initialize_weights(self):
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def _validate_input(self, data):
        if not torch.is_tensor(data) or data.ndim != 3:
            raise ValueError("Input must be a tensor shaped (B, N, 3)")
        if data.shape[-1] != self.space_dim:
            raise ValueError(f"Expected {self.space_dim} input channels")

    def _encode(self, coords):
        fx = self.preprocess(coords)
        return fx + self.placeholder.to(dtype=fx.dtype)[None, None, :]

    def _forward_block_range(self, fx, start, end, attn_kwargs=None):
        for block_index in range(start, end):
            fx = self.blocks[block_index](fx, attn_kwargs=attn_kwargs)
        return fx

    def _forward_blocks(self, fx, attn_kwargs=None):
        return self._forward_block_range(fx, 0, len(self.blocks), attn_kwargs)

    def _forward_with_checkpoint(self, fx, extra_tensor=None, kwargs_factory=None):
        for start in range(0, len(self.blocks), self.checkpoint_segment_size):
            end = min(start + self.checkpoint_segment_size, len(self.blocks))
            if extra_tensor is None:

                def run_segment(segment_fx, start=start, end=end):
                    return self._forward_block_range(segment_fx, start, end)

                fx = checkpoint(run_segment, fx, use_reentrant=False)
            else:

                def run_segment(segment_fx, tensor, start=start, end=end):
                    return self._forward_block_range(
                        segment_fx,
                        start,
                        end,
                        attn_kwargs=kwargs_factory(tensor),
                    )

                fx = checkpoint(
                    run_segment,
                    fx,
                    extra_tensor,
                    use_reentrant=False,
                )
        return fx

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        self._validate_input(data)
        fx = self._encode(data)
        if self.training:
            return self._forward_with_checkpoint(fx)
        return self._forward_blocks(fx)


LinearAttentionNeuralOperator = ClusterAttentionNeuralOperator

