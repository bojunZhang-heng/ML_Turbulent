import einops
import torch
import torch.nn.functional as F
import logging
import torch.nn as nn

from model.physics_attention import PhysicsAttention
from einops import rearrange, repeat

class SharedweightsSplitAttention(PhysicsAttention):
    def forward(
        self,
        x: torch.Tensor,
        split_size: list[int],
    ) -> torch.Tensor:
        """ Attention between:
        - q=surface_position -> kv=surface_position
        - q=volume_position -> kv=volume_position

        Args:

            x: Tensor containing positions (batch_size, num_points, pos_embed).
               pos_embed can be n_hideen
            split_size: How to split x into:
                len(split_size) == 2: (surface_queries, volume_queries)
        Returns:
            (batch size, num_points, n_hidden)
        """
        # B N C
        B, N, C = x.shape

        ### (1) Slice
        fx_mid = self.in_project_fx(x).reshape(B, N, self.n_heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C (batch_size, num_heads, num_points, dim_head)
        x_mid = self.in_project_x(x).reshape(B, N, self.n_heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C (batch_size, num_heads, num_points, dim_head)

        x_mid_split = x_mid.split(split_size, dim=2)
        x0_mid_split = x_mid_split[0]                    # Suface position
        x1_mid_split = x_mid_split[1]                    # Volume position

        fx_mid_split = fx_mid.split(split_size, dim=2)
        fx0_mid_split = fx_mid_split[0]
        fx1_mid_split = fx_mid_split[1]

        # surf+vol slice
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G (batch_size, num_heads, num_points, slice_token)
        slice_norm = slice_weights.sum(2)      # B H G (batch_size, num_heads, slice_token)
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        # surface slice
        slice_weights_0 = self.softmax(self.in_project_slice(x0_mid_split) / self.temperature)  # B H N G (batch_size, num_heads, num_points, slice_token)
        slice_norm_0 = slice_weights_0.sum(2)      # B H G (batch_size, num_heads, slice_token)
        slice_token_0 = torch.einsum("bhnc,bhng->bhgc", fx0_mid_split, slice_weights_0)
        slice_token_0 = slice_token_0 / ((slice_norm_0 + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        # volume slice
        slice_weights_1 = self.softmax(self.in_project_slice(x1_mid_split) / self.temperature)  # B H N G (batch_size, num_heads, num_points, slice_token)
        slice_norm_1 = slice_weights_1.sum(2)      # B H G (batch_size, num_heads, slice_token)
        slice_token_1 = torch.einsum("bhnc,bhng->bhgc", fx1_mid_split, slice_weights_1)
        slice_token_1 = slice_token_1 / ((slice_norm_1 + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens

        # for surface
        q_slice_token = self.to_q(slice_token)
        k_slice_token_0 = self.to_k(slice_token_0)
        v_slice_token_0 = self.to_v(slice_token_0)

        dots = torch.matmul(q_slice_token, k_slice_token_0.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token_0 = torch.matmul(attn, v_slice_token_0)  # B H G D (batch_size, num_heads, slice_token, dim_head)

        # for volume
        q_slice_token = self.to_q(slice_token)
        k_slice_token_1 = self.to_k(slice_token_1)
        v_slice_token_1 = self.to_v(slice_token_1)

        dots = torch.matmul(q_slice_token, k_slice_token_1.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token_1 = torch.matmul(attn, v_slice_token_1)  # B H G D (batch_size, num_heads, slice_token, dim_head)

        ### (3) Deslice

        # for surface
        out_x_0 = torch.einsum("bhgd,bhng->bhnd", out_slice_token_0, slice_weights_0)
        out_x_0 = rearrange(out_x_0, 'b h n d -> b n (h d)')     # 4*64 = 256

        # for volume
        out_x_1 = torch.einsum("bhgd,bhng->bhnd", out_slice_token_1, slice_weights_1)
        out_x_1 = rearrange(out_x_1, 'b h n d -> b n (h d)')     # 4*64 = 256

        out_x = torch.concat([out_x_0, out_x_1], dim=1)
        return self.to_out(out_x)


