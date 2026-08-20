import math
import torch
import torch.nn as nn


def _alibi_slopes(n_heads: int, device=None, dtype=None):
    """
    Slopes for ALiBi (Press et al., 2021).
    Each head gets a different slope, roughly spaced in log scale.
    """
    def get_slopes_power_of_2(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n_heads).is_integer():
        slopes = get_slopes_power_of_2(n_heads)
    else:
        closest_pow2 = 2 ** math.floor(math.log2(n_heads))
        slopes = get_slopes_power_of_2(closest_pow2)
        slopes_extra = get_slopes_power_of_2(2 * closest_pow2)
        slopes += slopes_extra[0::2][: n_heads - closest_pow2]
    slopes = torch.tensor(slopes, device=device, dtype=dtype)
    return slopes


def get_slopes(n):
    def get_slopes_power_of_2(n):
        start = (2 ** (-2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                           :n - closest_power_of_2]

class ALiBi2D(nn.Module):
    """
    2D ALiBi: per-head slopes for X and Y, build a bias for attention logits.
    If directional=False, uses |Δx|, |Δy| (distance-like); if True, uses signed deltas.
    final bias (added to logits) = -scale * (s_x[h]*comp_x + s_y[h]*comp_y)
    where comp_* is |Δ*| or signed Δ*.
    """
    def __init__(self, n_heads: int, directional: bool = False, scale: float = 1.0):
        super().__init__()
        self.directional = directional
        self.scale = scale
        # independent slopes per axis; you can share them if you prefer
        self.register_buffer("sx", _alibi_slopes(n_heads))
        self.register_buffer("sy", _alibi_slopes(n_heads))

    def forward(self, q_pos: torch.Tensor, k_pos: torch.Tensor) -> torch.Tensor:
        """
        q_pos: (B, Tq, 2) in [0,1]
        k_pos: (B, Tk, 2) in [0,1]
        returns: bias mask shaped for nn.MultiheadAttention:
                 (B * n_heads, Tq, Tk)
        """
        assert q_pos.dim() == 3 and q_pos.size(-1) == 2
        assert k_pos.dim() == 3 and k_pos.size(-1) == 2
        B, Tq, _ = q_pos.shape
        Tk = k_pos.shape[1]

        dx = q_pos[:, :, None, 0] - k_pos[:, None, :, 0]   # (B, Tq, Tk)
        dy = q_pos[:, :, None, 1] - k_pos[:, None, :, 1]   # (B, Tq, Tk)

        if not self.directional:
            dx = dx.abs()
            dy = dy.abs()

        # (1, H, 1, 1) broadcast across batch and timesteps
        H = self.sx.numel()
        sx = self.sx.view(1, H, 1, 1).to(q_pos)
        sy = self.sy.view(1, H, 1, 1).to(q_pos)

        # expand deltas to head dimension: (B, 1, Tq, Tk) -> (B, H, Tq, Tk)
        dx = dx.unsqueeze(1)
        dy = dy.unsqueeze(1)

        # Negative sign so larger deltas reduce logits (favoring nearby coords)
        bias = -(sx * dx + sy * dy) * self.scale              # (B, H, Tq, Tk)
        bias = bias.reshape(B * H, Tq, Tk)                    # for PyTorch MHA
        return bias


def pad_coords_for_cls(
        pos: torch.Tensor,  # (B,T,2)
        target_len: int,
        where: str = "prepend",  # or "append"
        strategy: str = "center",  # "center" | "mean" | "copy_first" | "copy_last"
):
    B, T, C = pos.shape
    if T == target_len:
        return pos
    if T == target_len - 1:
        if strategy == "center":
            cls = torch.full((B, 1, C), 0.5, device=pos.device, dtype=pos.dtype)
        elif strategy == "mean":
            cls = pos.mean(dim=1, keepdim=True)  # (B,1,2)
        elif strategy == "copy_first":
            cls = pos[:, :1, :]
        elif strategy == "copy_last":
            cls = pos[:, -1:, :]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return torch.cat([cls, pos], dim=1) if where == "prepend" else torch.cat([pos, cls], dim=1)

    raise RuntimeError(f"Coords length {T} doesn’t match target {target_len} (and is not off by 1).")



def pad_coords_for_specials(
        pos: torch.Tensor,  # (B,T,2)
        target_len: int,
        where: str = "prepend",  # or "append"
        strategy: str = "center",  # "center" | "mean" | "copy_first" | "copy_last"
):
    B, T, C = pos.shape
    num_to_pad = target_len - T

    if num_to_pad == 0:
        return pos

    if num_to_pad > 0:
        if strategy == "center":
            specials = torch.full((B, num_to_pad, C), 0.5, device=pos.device, dtype=pos.dtype)
        elif strategy == "mean":
            specials = pos.mean(dim=1, keepdim=True).expand(-1, num_to_pad, -1)
        elif strategy == "copy_first":
            specials = specials = pos[:, :1, :].expand(-1, num_to_pad, -1)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return torch.cat([specials, pos], dim=1) if where == "prepend" else torch.cat([pos, specials], dim=1)

    raise RuntimeError(f"Coords length {T} doesn’t match target {target_len} (and is not off by 1).")