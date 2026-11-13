import torch


def _right_shift(binary, shift):
    if binary.shape[-1] <= shift:
        return torch.zeros_like(binary)
    return torch.nn.functional.pad(binary[..., :-shift], (shift, 0), value=0)


def _gray_to_binary(gray):
    shift = 1 << max(0, (gray.shape[-1] - 1).bit_length() - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, _right_shift(gray, shift))
        shift //= 2
    return gray


def _hilbert_encode(locations, num_bits):
    num_dims = 3
    if num_dims * num_bits > 63:
        raise ValueError("Hilbert code exceeds the signed int64 capacity")

    bit_mask = 1 << torch.arange(8, device=locations.device)
    bit_mask_reversed = bit_mask.flip(-1)
    locations_uint8 = (
        locations.long().contiguous().view(torch.uint8).reshape(-1, num_dims, 8).flip(-1)
    )
    gray = (
        locations_uint8.unsqueeze(-1)
        .bitwise_and(bit_mask_reversed)
        .ne(0)
        .byte()
        .flatten(-2, -1)[..., -num_bits:]
    )

    for bit in range(num_bits):
        for dim in range(num_dims):
            mask = gray[:, dim, bit]
            gray[:, 0, bit + 1:] = torch.logical_xor(
                gray[:, 0, bit + 1:], mask[:, None]
            )
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).expand(-1, gray.shape[2] - bit - 1),
                torch.logical_xor(
                    gray[:, 0, bit + 1:], gray[:, dim, bit + 1:]
                ),
            )
            gray[:, dim, bit + 1:] = torch.logical_xor(
                gray[:, dim, bit + 1:], to_flip
            )
            gray[:, 0, bit + 1:] = torch.logical_xor(
                gray[:, 0, bit + 1:], to_flip
            )

    gray = gray.swapaxes(1, 2).reshape(-1, num_bits * num_dims)
    binary = _gray_to_binary(gray)
    padded = torch.nn.functional.pad(binary, (64 - num_bits * num_dims, 0), value=0)
    encoded_uint8 = (
        (padded.flip(-1).reshape(-1, 8, 8) * bit_mask)
        .sum(2)
        .to(torch.uint8)
        .contiguous()
    )
    return encoded_uint8.view(torch.int64).reshape(-1)


def _morton_encode(locations, num_bits):
    locations = locations.long()
    code = torch.zeros(locations.shape[0], device=locations.device, dtype=torch.long)
    for bit in range(num_bits):
        code |= ((locations[:, 0] >> bit) & 1) << (3 * bit + 2)
        code |= ((locations[:, 1] >> bit) & 1) << (3 * bit + 1)
        code |= ((locations[:, 2] >> bit) & 1) << (3 * bit)
    return code


@torch.inference_mode()
def pos_to_order_inverse_index(pos, device, tensor=False, grid_size=0.05):
    """Build four locality-preserving orders from the first three coordinates."""
    if pos.ndim == 2:
        pos = pos.unsqueeze(0)
    if pos.ndim != 3 or pos.shape[-1] != 3:
        raise ValueError(f"Expected positions shaped (B, N, 3), got {tuple(pos.shape)}")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    orders = []
    inverses = []
    for coordinates in pos:
        coordinates = coordinates.to(device=device, dtype=torch.float32)
        grid = torch.div(
            coordinates - coordinates.amin(dim=0),
            grid_size,
            rounding_mode="trunc",
        ).to(torch.long)
        depth = max(1, int(grid.max().item()).bit_length())
        if depth > 16:
            raise ValueError(
                f"Serialization depth {depth} exceeds 16; increase grid_size"
            )

        transposed = grid[:, [1, 0, 2]]
        codes = torch.stack(
            [
                _morton_encode(grid, depth),
                _morton_encode(transposed, depth),
                _hilbert_encode(grid, depth),
                _hilbert_encode(transposed, depth),
            ],
            dim=0,
        )
        order = torch.argsort(codes, dim=1)
        inverse = torch.empty_like(order).scatter_(
            1,
            order,
            torch.arange(order.shape[1], device=device).expand_as(order),
        )
        orders.append(order)
        inverses.append(inverse)

    order = torch.stack(orders, dim=0)
    inverse = torch.stack(inverses, dim=0)
    if not tensor:
        return order.cpu().numpy(), inverse.cpu().numpy()
    return order, inverse
