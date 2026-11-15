"""Minimal coordinate serialization used by the transient SATO model."""

from __future__ import annotations

import torch

from SATO_dataset.serialization import encode


@torch.no_grad()
def pos_to_order_inverse_index(pos, device, tensor=False, grid_size=0.05):
    if pos.dim() == 2:
        pos = pos.unsqueeze(0)
    if pos.ndim != 3 or pos.shape[-1] != 3:
        raise ValueError(f"Expected positions shaped (B,N,3), got {tuple(pos.shape)}")

    orders, inverses = [], []
    order_names = ("z", "z-trans", "hilbert", "hilbert-trans")
    for batch_index in range(pos.shape[0]):
        coordinates = pos[batch_index].to(device)
        grid = torch.div(
            coordinates - coordinates.amin(dim=0),
            grid_size,
            rounding_mode="trunc",
        ).int()
        depth = max(1, int(grid.max()).bit_length())
        if depth > 16:
            raise ValueError(
                f"Serialization depth {depth} exceeds 16; increase grid_size"
            )
        batch = torch.zeros(len(grid), device=device, dtype=torch.long)
        codes = torch.stack(
            [encode(grid, batch, depth, order=name) for name in order_names]
        )
        order = torch.argsort(codes, dim=1)
        inverse = torch.empty_like(order).scatter_(
            1,
            order,
            torch.arange(len(grid), device=device).repeat(len(order_names), 1),
        )
        orders.append(order.clone())
        inverses.append(inverse.clone())

    order = torch.stack(orders)
    inverse = torch.stack(inverses)
    if tensor:
        return order, inverse
    return order.cpu().numpy(), inverse.cpu().numpy()
