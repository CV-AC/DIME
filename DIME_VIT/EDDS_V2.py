import torch
import torch.nn.functional as F


def _dc_free_delta(patches: torch.Tensor) -> torch.Tensor:

    B, L, D = patches.shape
    assert D % 3 == 0, f"Expected RGB patch dimension, got {D}"
    x = patches.reshape(B, L, D // 3, 3)
    x = x - x.mean(dim=-2, keepdim=True)
    return x.reshape(B, L, D)


def _expand_mask(
    mask: torch.Tensor, batch_size: int, dtype: torch.dtype
) -> torch.Tensor:

    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1, -1)
    return mask.squeeze(-1).to(dtype=dtype)


def compute_edds_v2_loss(
    *,
    target_rgb: torch.Tensor,
    unmix_rgb: torch.Tensor,
    mask: torch.Tensor,
    norm_pix_loss: bool,
    p_std: torch.Tensor = None,
    p_mean: torch.Tensor = None,
    sobel_q: float = 0.5,
    masked_weight: float = 1.0,
    visible_weight: float = 0.25,
    gt_weight_clip: float = 3.0,
    pair_weight_clip: float = 2.0,
    pred_mag_tau: float = 0.05,
    pred_weight_floor: float = 0.10,
    norm_eps: float = 0.03,
    eps: float = 1e-6,
) -> torch.Tensor:

    B, L, _ = target_rgb.shape
    dtype = unmix_rgb.dtype

    with torch.no_grad():
        delta_gt_raw = target_rgb - target_rgb.flip(0)
        delta_gt_ac = _dc_free_delta(delta_gt_raw)
        delta_gt_mag = delta_gt_ac.norm(dim=-1)
        gt_delta_dir = F.normalize(delta_gt_ac, dim=-1, eps=eps)

        per_image_ref = delta_gt_mag.mean(dim=-1, keepdim=True).clamp_min(eps)
        gt_weight = (delta_gt_mag / per_image_ref).clamp(max=gt_weight_clip)

        K = max(1, int(round(L * (1.0 - sobel_q))))
        kth = delta_gt_mag.topk(K, dim=-1).values[..., -1:].detach()
        softness = (0.10 * per_image_ref).clamp_min(eps)
        soft_topk = torch.sigmoid((delta_gt_mag - kth) / softness)
        gt_weight = gt_weight * (0.25 + 0.75 * soft_topk)

        pair_mag = delta_gt_mag.mean(dim=-1, keepdim=True)
        batch_ref = pair_mag.mean().detach().clamp_min(eps)
        pair_weight = (pair_mag / batch_ref).clamp(max=pair_weight_clip)

        mask_w = _expand_mask(mask, B, dtype=delta_gt_mag.dtype)
        spatial_weight = visible_weight + (masked_weight - visible_weight) * mask_w

        base_weight = (gt_weight * pair_weight * spatial_weight).to(dtype=dtype)
        gt_delta_dir = gt_delta_dir.to(dtype=dtype)

    if norm_pix_loss and p_std is not None and p_mean is not None:
        pred_raw = unmix_rgb * p_std + p_mean
        pair_raw = unmix_rgb.flip(0) * p_std.flip(0) + p_mean.flip(0)
        delta_pred = pred_raw - pair_raw
    else:
        delta_pred = unmix_rgb - unmix_rgb.flip(0)

    delta_pred_ac = _dc_free_delta(delta_pred)

    pred_mag = delta_pred_ac.norm(dim=-1).detach()
    pred_weight = pred_weight_floor + (1.0 - pred_weight_floor) * (
        pred_mag / (pred_mag + pred_mag_tau)
    )
    weight = base_weight * pred_weight.to(dtype=dtype)

    norm = (delta_pred_ac.norm(dim=-1, keepdim=True).pow(2) + norm_eps**2).sqrt()
    cos_d = (delta_pred_ac / norm * gt_delta_dir).sum(dim=-1)

    denom = weight.sum().clamp_min(1.0)
    return ((1.0 - cos_d) * weight).sum() / denom
