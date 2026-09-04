import einops
import torch
import torch.nn.functional as F
import logging
import torch.nn as nn

from model.physics_attention import PhysicsAttention
from einops import rearrange, repeat
from modules.rope import rope

class PerceiverAttention(PhysicsAttention):
    def forward(
        self,
        q_surf: torch.Tensor,
        q_volume: torch.Tensor,
        k_geometry: torch.Tensor,
        v_geometry: torch.Tensor,
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
        B, N, C = q_surf.shape
        _, N_volume, _ = q_volume.shape

        ### (1) Slice
        # Geometry
        fx_mid_g = self.in_project_fx(k_geometry).reshape(B, N, self.n_heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C (batch_size, num_heads, num_points, dim_head)
        x_mid_g = self.in_project_x(k_geometry).reshape(B, N, self.n_heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C (batch_size, num_heads, num_points, dim_head)

        slice_weights_g = self.softmax(self.in_project_slice(x_mid_g) / self.temperature)  # B H N G (batch_size, num_heads, num_points, slice_token)
        slice_norm_g = slice_weights_g.sum(2)      # B H G (batch_size, num_heads, slice_token)
        slice_token_g = torch.einsum("bhnc,bhng->bhgc", fx_mid_g, slice_weights_g)
        slice_token_g = slice_token_g / ((slice_norm_g + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        # Surface
        fx_mid_surf = self.in_project_fx(q_surf).reshape(B, N, self.n_heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C (batch_size, num_heads, num_points, dim_head)
        x_mid_surf = self.in_project_x(q_surf).reshape(B, N, self.n_heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C (batch_size, num_heads, num_points, dim_head)

        slice_weights_surf = self.softmax(self.in_project_slice(x_mid_surf) / self.temperature)  # B H N G (batch_size, num_heads, num_points, slice_token)
        slice_norm_surf = slice_weights_surf.sum(2)      # B H G (batch_size, num_heads, slice_token)
        slice_token_surf = torch.einsum("bhnc,bhng->bhgc", fx_mid_surf, slice_weights_surf)
        slice_token_surf = slice_token_surf / ((slice_norm_surf + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        # Volume
        fx_mid_volume = self.in_project_fx(q_volume).reshape(B, N_volume, self.n_heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C (batch_size, num_heads, num_points, dim_head)
        x_mid_volume = self.in_project_x(q_volume).reshape(B, N_volume, self.n_heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C (batch_size, num_heads, num_points, dim_head)

        slice_weights_volume = self.softmax(self.in_project_slice(x_mid_volume) / self.temperature)  # B H N G (batch_size, num_heads, num_points, slice_token)
        slice_norm_volume = slice_weights_volume.sum(2)      # B H G (batch_size, num_heads, slice_token)
        slice_token_volume = torch.einsum("bhnc,bhng->bhgc", fx_mid_volume, slice_weights_volume)
        slice_token_volume = slice_token_volume / ((slice_norm_volume + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens

        # for surface
        q_slice_token_surf = self.to_q(slice_token_surf)
        k_slice_token_g = self.to_k(slice_token_g)
        v_slice_token_g = self.to_v(slice_token_g)

        dots = torch.matmul(q_slice_token_surf, k_slice_token_g.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token_surf = torch.matmul(attn, v_slice_token_g)  # B H G D (batch_size, num_heads, slice_token, dim_head)

        # for volume
        q_slice_token_volume = self.to_q(slice_token_volume)

        dots = torch.matmul(q_slice_token_volume, k_slice_token_g.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token_volume = torch.matmul(attn, v_slice_token_g)  # B H G D (batch_size, num_heads, slice_token, dim_head)

        ### (3) Deslice

        # for surface
        out_x_surf = torch.einsum("bhgd,bhng->bhnd", out_slice_token_surf, slice_weights_surf)
        out_x_surf = rearrange(out_x_surf, 'b h n d -> b n (h d)')     # 4*64 = 256

        # for volume
        out_x_volume = torch.einsum("bhgd,bhng->bhnd", out_slice_token_volume, slice_weights_volume)
        out_x_volume = rearrange(out_x_volume, 'b h n d -> b n (h d)')     # 4*64 = 256

        #logging.info(f"out_x_volume.shape: {out_x_volume.shape}")
        #logging.info(f"out_x_surf.shape: {out_x_surf.shape}")

        return self.to_out(out_x_surf), self.to_out(out_x_volume)



