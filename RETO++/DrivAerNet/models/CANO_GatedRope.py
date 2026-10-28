import math

import torch
import torch.nn.functional as F

from models.RETO_plus import (
    RETOPlusAttention,
    Model,
    apply_rotary_embedding,
)


class GatedRopeClusterAttention(RETOPlusAttention):
    """CANO with a learnable convex interpolation of base and RoPE logits."""

    def __init__(self, *args, initial_rope_gate=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0.0 < initial_rope_gate < 1.0:
            raise ValueError("initial_rope_gate must be between 0 and 1")
        gate_logit = math.log(initial_rope_gate / (1.0 - initial_rope_gate))
        self.rope_gate_logit = torch.nn.Parameter(
            torch.full((1, self.heads, 1, 1), gate_logit)
        )

    def forward(self, x, position_phases):
        q, k, v = self._project_qkv(x)
        cluster_assignment = self._cluster_assignment(q)
        clustered_q = self._aggregate_cluster_queries(cluster_assignment, q)

        q_rope = apply_rotary_embedding(q, position_phases)
        k_rope = apply_rotary_embedding(k, position_phases)
        clustered_q_rope = self._aggregate_cluster_queries(
            cluster_assignment, q_rope
        )

        gate = torch.sigmoid(self.rope_gate_logit).to(dtype=q.dtype)
        combined_q = torch.cat(
            (
                torch.sqrt(1.0 - gate) * clustered_q,
                torch.sqrt(gate) * clustered_q_rope,
            ),
            dim=-1,
        )
        combined_k = torch.cat(
            (
                torch.sqrt(1.0 - gate) * k,
                torch.sqrt(gate) * k_rope,
            ),
            dim=-1,
        )
        cluster_features = F.scaled_dot_product_attention(
            combined_q,
            combined_k,
            v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scale,
        )
        output = torch.einsum(
            "bhng,bhgd->bhnd", cluster_assignment, cluster_features
        )
        output = output.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.to_out(output)


class GatedRopeClusterAttentionNeuralOperator(
    Model
):
    """CANO retaining the base logits plus a gated three-dimensional RoPE path."""

    def __init__(self, *args, initial_rope_gate=0.05, **kwargs):
        super().__init__(
            *args,
            _attention_cls=GatedRopeClusterAttention,
            _attention_kwargs={"initial_rope_gate": initial_rope_gate},
            **kwargs,
        )

    def rope_gate_values(self) -> torch.Tensor:
        return torch.stack(
            [
                torch.sigmoid(block.attn.rope_gate_logit).flatten()
                for block in self.blocks
            ]
        )


LinearAttentionNeuralOperator = GatedRopeClusterAttentionNeuralOperator

