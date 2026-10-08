from __future__ import annotations

import math
import random
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


GridSize = Tuple[int, int]


def _to_2tuple(value) -> GridSize:
    return (value, value) if isinstance(value, int) else tuple(value)


def _drop_path(x: torch.Tensor, probability: float, training: bool) -> torch.Tensor:
    if probability == 0.0 or not training:
        return x
    keep = 1.0 - probability
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    return (
        x * torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(keep) / keep
    )


def _sincos_2d(
    dim: int, grid_size: GridSize, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:

    if dim % 4:
        raise ValueError(
            f"2-D sine/cosine embedding needs dim divisible by 4, got {dim}"
        )
    h, w = grid_size
    axis_dim = dim // 2
    omega = torch.arange(axis_dim // 2, device=device, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (omega / (axis_dim / 2)))
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    y = yy.reshape(-1, 1) * omega.reshape(1, -1)
    x = xx.reshape(-1, 1) * omega.reshape(1, -1)
    pos = torch.cat((y.sin(), y.cos(), x.sin(), x.cos()), dim=-1)
    return pos.to(dtype=dtype).unsqueeze(0)


def _rope_2d(
    head_dim: int, grid_size: GridSize, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:

    if head_dim % 4:
        raise ValueError(f"2-D RoPE needs head_dim divisible by 4, got {head_dim}")
    h, w = grid_size
    axis_dim = head_dim // 2
    omega = torch.arange(0, axis_dim, 2, device=device, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (omega / axis_dim))
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    y = yy.reshape(-1, 1) * omega.reshape(1, -1)
    x = xx.reshape(-1, 1) * omega.reshape(1, -1)
    cos = torch.cat(
        (y.cos().repeat_interleave(2, -1), x.cos().repeat_interleave(2, -1)), -1
    )
    sin = torch.cat(
        (y.sin().repeat_interleave(2, -1), x.sin().repeat_interleave(2, -1)), -1
    )
    return cos.to(dtype)[None, None], sin.to(dtype)[None, None]


def _rotate_axial_pairs(x: torch.Tensor) -> torch.Tensor:

    y, x_axis = x.chunk(2, dim=-1)

    def rotate(part: torch.Tensor) -> torch.Tensor:
        shape = part.shape
        part = part.reshape(*shape[:-1], -1, 2)
        first, second = part.unbind(-1)
        return torch.stack((-second, first), dim=-1).reshape(shape)

    return torch.cat((rotate(y), rotate(x_axis)), dim=-1)


class PatchEmbed(nn.Module):

    def __init__(self, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride=patch_size)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, GridSize]:
        x = self.proj(images)
        grid_size = (x.shape[-2], x.shape[-1])
        return x.flatten(2).transpose(1, 2), grid_size


class GeGLU(nn.Module):

    def __init__(self, dim: int, mlp_ratio: float, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio * 2.0 / 3.0)
        hidden = max(8, int(round(hidden / 8)) * 8)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.fc1(x).chunk(2, dim=-1)
        x = F.gelu(gate) * value
        return self.drop(self.fc2(self.drop(x)))


class Attention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        gated: bool = False,
        gate_init_bias: float = 2.0,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.gate = nn.Linear(dim, dim) if gated else None

        self.gate_init_bias = float(gate_init_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.reset_gate_parameters()

    def reset_gate_parameters(self) -> None:

        if self.gate is not None:
            nn.init.trunc_normal_(self.gate.weight, std=0.02)
            nn.init.zeros_(self.gate.bias)

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):

        dropout = self.attn_drop if self.training else 0.0
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        if mask is not None:
            mask = mask.contiguous()
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        source_layout: Optional[Tuple[torch.Tensor, torch.Tensor, int]] = None,
        rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        batch, length, dim = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        if rope is not None:
            cos, sin = rope
            q = q * cos + _rotate_axial_pairs(q) * sin
            k = k * cos + _rotate_axial_pairs(k) * sin

        if source_mask is None:
            out = self._sdpa(q, k, v)
        elif source_layout is not None:

            order, inverse, split = source_layout
            gather = order[:, None, :, None].expand(
                -1, self.num_heads, -1, self.head_dim
            )
            q, k, v = q.gather(2, gather), k.gather(2, gather), v.gather(2, gather)
            pieces = []
            if split:
                pieces.append(
                    self._sdpa(q[:, :, :split], k[:, :, :split], v[:, :, :split])
                )
            if split < length:
                pieces.append(
                    self._sdpa(q[:, :, split:], k[:, :, split:], v[:, :, split:])
                )
            out = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=2)
            restore = inverse[:, None, :, None].expand(
                -1, self.num_heads, -1, self.head_dim
            )
            out = out.gather(2, restore)
        else:

            mask = source_mask
            if mask.shape[0] == 1:
                mask = mask.expand(batch, -1, -1)
            group = mask.squeeze(-1) > 0.5
            allow = group[:, None, :, None] == group[:, None, None, :]
            out = self._sdpa(q, k, v, allow)

        if self.gate is not None:
            gate = self.gate(x).reshape(batch, length, self.num_heads, self.head_dim)
            gate = gate.permute(0, 2, 1, 3)
            out = out * gate.to(dtype=out.dtype)
        out = out.transpose(1, 2).reshape(batch, length, dim)
        return self.proj_drop(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        gated_attention: bool = False,
        gate_init_bias: float = 2.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads,
            qkv_bias,
            attn_drop,
            drop,
            gated=gated_attention,
            gate_init_bias=gate_init_bias,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = GeGLU(dim, mlp_ratio, drop)
        self.drop_path = drop_path

    def forward(self, x, source_mask=None, source_layout=None, rope=None):
        x = x + _drop_path(
            self.attn(self.norm1(x), source_mask, source_layout, rope),
            self.drop_path,
            self.training,
        )
        return x + _drop_path(self.mlp(self.norm2(x)), self.drop_path, self.training)


class SourceAlignedPool(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=2, stride=2, groups=dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        nn.init.constant_(self.conv.weight, 0.25)

    def forward(self, tokens: torch.Tensor, grid_size: GridSize) -> torch.Tensor:
        batch, _, dim = tokens.shape
        h, w = grid_size
        x = tokens.transpose(1, 2).reshape(batch, dim, h, w)
        x = self.conv(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class OrientedGradientExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
        self.register_buffer("weight_x", kx.reshape(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer("weight_y", ky.reshape(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gx = F.conv2d(images, self.weight_x, padding=1, groups=3)
        gy = F.conv2d(images, self.weight_y, padding=1, groups=3)
        return gx, gy


def _dc_free_delta(patches: torch.Tensor) -> torch.Tensor:
    batch, length, channels = patches.shape
    x = patches.reshape(batch, length, channels // 3, 3)
    return (x - x.mean(dim=-2, keepdim=True)).reshape_as(patches)


def _edds_v1(
    target_rgb: torch.Tensor,
    unmix_rgb: torch.Tensor,
    norm_pix_loss: bool,
    p_std: Optional[torch.Tensor],
    p_mean: Optional[torch.Tensor],
    sobel_q: float,
) -> torch.Tensor:
    length = target_rgb.shape[1]
    with torch.no_grad():
        delta_gt = _dc_free_delta(target_rgb - target_rgb.flip(0))
        delta_mag = delta_gt.norm(dim=-1)
        pair_mag = delta_mag.mean(dim=-1)
        totals = torch.stack((pair_mag.sum(), pair_mag.new_tensor(pair_mag.numel())))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals)
        batch_mean = totals[0] / totals[1].clamp_min(1.0)
        topk = min(length, max(0, int(round(length * (1.0 - sobel_q)))))
        indices = delta_mag.topk(topk, dim=-1).indices
        support = torch.zeros_like(delta_mag).scatter_(-1, indices, 1.0)
        support = support * (pair_mag > 0.2 * batch_mean)[:, None]

    if norm_pix_loss:
        if p_std is None or p_mean is None:
            raise ValueError("EDDS V1 requires target patch statistics")
        prediction = unmix_rgb * p_std.detach() + p_mean.detach()
    else:
        prediction = unmix_rgb
    delta_pred = _dc_free_delta(prediction - prediction.flip(0))
    numerator = (delta_pred * delta_gt).sum(dim=-1)
    denominator = delta_pred.norm(dim=-1) * delta_mag + 1e-6
    discrepancy = 1.0 - numerator / denominator
    per_pair = (discrepancy * support).sum(dim=-1) / support.sum(dim=-1).clamp_min(1.0)
    return per_pair.mean()


def _edds_v2(
    target_rgb: torch.Tensor,
    unmix_rgb: torch.Tensor,
    mask: torch.Tensor,
    norm_pix_loss: bool,
    p_std: Optional[torch.Tensor],
    p_mean: Optional[torch.Tensor],
    sobel_q: float,
) -> torch.Tensor:

    batch, length, _ = target_rgb.shape
    dtype = unmix_rgb.dtype
    eps = 1e-6

    with torch.no_grad():
        delta_gt = _dc_free_delta(target_rgb - target_rgb.flip(0))
        delta_mag = delta_gt.norm(dim=-1)
        delta_dir = F.normalize(delta_gt, dim=-1, eps=eps)
        image_ref = delta_mag.mean(dim=-1, keepdim=True).clamp_min(eps)
        gt_weight = (delta_mag / image_ref).clamp(max=3.0)
        topk = max(1, int(round(length * (1.0 - sobel_q))))
        threshold = delta_mag.topk(topk, dim=-1).values[..., -1:].detach()
        soft_topk = torch.sigmoid(
            (delta_mag - threshold) / (0.10 * image_ref).clamp_min(eps)
        )
        gt_weight = gt_weight * (0.25 + 0.75 * soft_topk)
        pair_mag = delta_mag.mean(dim=-1, keepdim=True)
        pair_weight = (pair_mag / pair_mag.mean().detach().clamp_min(eps)).clamp(
            max=2.0
        )
        if mask.shape[0] == 1:
            mask = mask.expand(batch, -1, -1)
        spatial_weight = 0.25 + 0.75 * mask.squeeze(-1)
        base_weight = (gt_weight * pair_weight * spatial_weight).to(dtype)
        delta_dir = delta_dir.to(dtype)

    if norm_pix_loss and p_std is not None and p_mean is not None:
        prediction = unmix_rgb * p_std + p_mean
        paired = unmix_rgb.flip(0) * p_std.flip(0) + p_mean.flip(0)
        delta_pred = prediction - paired
    else:
        delta_pred = unmix_rgb - unmix_rgb.flip(0)
    delta_pred = _dc_free_delta(delta_pred)
    pred_mag = delta_pred.norm(dim=-1).detach()
    pred_weight = 0.10 + 0.90 * pred_mag / (pred_mag + 0.05)
    weight = base_weight * pred_weight.to(dtype)
    norm = (delta_pred.norm(dim=-1, keepdim=True).square() + 0.03**2).sqrt()
    cosine = (delta_pred / norm * delta_dir).sum(dim=-1)
    return ((1.0 - cosine) * weight).sum() / weight.sum().clamp_min(1.0)


class DIMEViT(nn.Module):

    input_mean = (0.485, 0.456, 0.406)
    input_std = (0.229, 0.224, 0.225)

    def __init__(
        self,
        img_size=224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        decoder_dim: int = 384,
        decoder_depth: int = 6,
        decoder_num_heads: int = 16,
        mask_ratio: float = 0.5,
        mask_cell_size: int = 1,
        shared_mask: bool = True,
        range_mask_ratio: float = 0.0,
        mask_strategy: str = "single",
        mask_block_sizes: Optional[Sequence[int]] = None,
        block_probs: Optional[Sequence[float]] = None,
        pos_encoding: str = "rope",
        attention_mode: str = "auto",
        gated_attention: bool = False,
        gate_init_bias: float = 2.0,
        norm_pix_loss: bool = True,
        lambda_diff: float = 0.5,
        sobel_q: float = 0.5,
        lambda_edds: float = 1.0,
        edds_warmup_epochs: int = 20,
        edds_version: str = "v2",
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if patch_size != 16:
            raise ValueError(
                "DIME_VIT implements DeiT-style /16 backbones; patch_size must be 16"
            )
        if in_chans != 3:
            raise ValueError("RGB, Sobel, and EDDS heads require in_chans=3")
        if mask_cell_size not in (1, 2):
            raise ValueError("mask_cell_size must be 1 (16px) or 2 (32px)")
        if pos_encoding not in ("rope", "sincos"):
            raise ValueError("pos_encoding must be 'rope' or 'sincos'")
        if attention_mode not in ("auto", "sorted", "dense"):
            raise ValueError("attention_mode must be 'auto', 'sorted', or 'dense'")
        if mask_strategy not in ("single", "multiscale"):
            raise ValueError("mask_strategy must be 'single' or 'multiscale'")
        if not 0.0 <= mask_ratio <= 1.0 or not 0.0 <= sobel_q <= 1.0:
            raise ValueError("mask_ratio and sobel_q must lie in [0, 1]")
        if not range_mask_ratio >= 0.0:
            raise ValueError("range_mask_ratio must be non-negative")
        if not gate_init_bias >= 0.0:
            raise ValueError("gate_init_bias must be non-negative")

        self.img_size = _to_2tuple(img_size)
        self.patch_size = patch_size
        self.reconstruction_patch_size = patch_size * mask_cell_size
        self.mask_cell_size = mask_cell_size
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.decoder_dim = decoder_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.encoder_stride = self.reconstruction_patch_size
        self.mask_ratio = mask_ratio
        self.shared_mask = shared_mask
        self.range_mask_ratio = range_mask_ratio
        self.mask_strategy = mask_strategy
        self.pos_encoding = pos_encoding
        self.attention_mode = attention_mode
        self.gated_attention = bool(gated_attention)
        self.gate_init_bias = float(gate_init_bias)
        self.norm_pix_loss = norm_pix_loss
        self.lambda_diff = lambda_diff
        self.sobel_q = sobel_q
        self.lambda_edds = lambda_edds
        self.edds_warmup_epochs = edds_warmup_epochs
        self.edds_version = edds_version.lower()
        self.use_checkpoint = use_checkpoint

        if self.edds_version not in ("v1", "v2"):
            raise ValueError("edds_version must be 'v1' or 'v2'")
        configured_blocks = list(mask_block_sizes or [1])
        if any(
            float(size) < 1 or not float(size).is_integer()
            for size in configured_blocks
        ):
            raise ValueError("mask_block_sizes must contain positive integers")
        self.mask_block_sizes = [int(size) for size in configured_blocks]
        self.block_probs = (
            None if block_probs is None else [float(value) for value in block_probs]
        )
        if self.block_probs is not None and len(self.block_probs) != len(
            self.mask_block_sizes
        ):
            raise ValueError("block_probs must match mask_block_sizes")
        if self.block_probs is not None and (
            any(not value >= 0.0 for value in self.block_probs)
            or not any(self.block_probs)
        ):
            raise ValueError("block_probs must be non-negative with a positive sum")

        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias,
                    drop_rate,
                    attn_drop_rate,
                    rates[index],
                    gated_attention=self.gated_attention,
                    gate_init_bias=self.gate_init_bias,
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.pos_drop = nn.Dropout(drop_rate)

        if mask_cell_size == 2:
            self.decoder_pool = SourceAlignedPool(embed_dim)
        else:
            self.decoder_pool = None

        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    decoder_dim, decoder_num_heads, 4.0, True, 0.0, 0.0, 0.0
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim, eps=1e-6)
        output_pixels = self.reconstruction_patch_size**2
        self.decoder_pred = nn.Linear(decoder_dim, output_pixels * 3)
        self.decoder_pred_diff = nn.Linear(decoder_dim, output_pixels * 6)
        self.sobel = OrientedGradientExtractor()

        self.apply(self._init_weights)

        for block in self.blocks:
            block.attn.reset_gate_parameters()
        nn.init.normal_(self.mask_token, std=0.02)
        if self.decoder_pool is not None:
            nn.init.constant_(self.decoder_pool.conv.weight, 0.25)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight.flatten(1))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _validate_images(self, images: torch.Tensor, paired: bool) -> GridSize:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"expected images [B, 3, H, W], got {tuple(images.shape)}")
        if paired and images.shape[0] % 2:
            raise ValueError(
                "DIME mix/unmix requires an even batch arranged as flip-aligned pairs"
            )
        h, w = images.shape[-2:]

        stride = self.reconstruction_patch_size if paired else self.patch_size
        if h % stride or w % stride:
            requirement = (
                f"16*mask_cell_size={stride}" if paired else f"patch_size={stride}"
            )
            raise ValueError(
                f"H and W must be divisible by {requirement}, got {(h, w)}"
            )
        return h // self.patch_size, w // self.patch_size

    def _sample_mask(
        self, images: torch.Tensor, mask_ratio: Optional[float]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[int], GridSize, GridSize]:
        batch, _, height, width = images.shape
        encoder_grid = (height // self.patch_size, width // self.patch_size)
        decoder_grid = (
            encoder_grid[0] // self.mask_cell_size,
            encoder_grid[1] // self.mask_cell_size,
        )
        ratio = self.mask_ratio if mask_ratio is None else float(mask_ratio)
        if self.range_mask_ratio:
            ratio += random.uniform(0.0, self.range_mask_ratio)
        ratio = min(max(ratio, 0.0), 1.0)
        num_masks = 1 if self.shared_mask else batch // 2
        dh, dw = decoder_grid

        if self.mask_strategy == "multiscale":

            weights = self.block_probs or [1.0] * len(self.mask_block_sizes)
            weight_sum = float(sum(weights))
            score = torch.zeros(num_masks, 1, dh, dw, device=images.device)
            for block_size, weight in zip(self.mask_block_sizes, weights):
                coarse_h = math.ceil(dh / block_size)
                coarse_w = math.ceil(dw / block_size)
                if coarse_h * coarse_w < 2:
                    raise ValueError(
                        f"mask block {block_size} is too large for decoder grid {decoder_grid}"
                    )
                field = torch.rand(
                    num_masks, 1, coarse_h, coarse_w, device=images.device
                )
                if (coarse_h, coarse_w) != decoder_grid:
                    field = F.interpolate(field, size=decoder_grid, mode="nearest")
                score.add_(field, alpha=float(weight) / weight_sum)

            length = dh * dw
            count = int(length * ratio)

            score = score.flatten(1) + torch.rand_like(score.flatten(1)) * 1.0e-6
            indices = score.argsort(dim=1)[:, :count]
            mask_2d = torch.zeros(num_masks, length, device=images.device)
            mask_2d.scatter_(1, indices, 1.0)
            mask_2d = mask_2d.reshape(num_masks, 1, dh, dw)
            split = length - count
        else:
            mask_2d, split = self._sample_single_scale_mask(
                num_masks, decoder_grid, ratio, images.device
            )

        if not self.shared_mask:
            pair_index = torch.tensor(
                [min(i, batch - 1 - i) for i in range(batch)], device=images.device
            )
            mask_2d = mask_2d[pair_index]

        decoder_mask = mask_2d.flatten(2).transpose(1, 2).contiguous()
        encoder_2d = mask_2d.repeat_interleave(self.mask_cell_size, -2)
        encoder_2d = encoder_2d.repeat_interleave(self.mask_cell_size, -1)
        encoder_mask = encoder_2d.flatten(2).transpose(1, 2).contiguous()
        if split is not None:
            split *= self.mask_cell_size**2
        return decoder_mask, encoder_mask, split, encoder_grid, decoder_grid

    def _sample_single_scale_mask(
        self,
        num_masks: int,
        decoder_grid: GridSize,
        ratio: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Optional[int]]:

        block_size = (
            self.mask_block_sizes[0]
            if len(self.mask_block_sizes) == 1
            else random.choices(self.mask_block_sizes, weights=self.block_probs, k=1)[0]
        )
        dh, dw = decoder_grid

        if block_size > 1:
            coarse_h, coarse_w = dh // block_size, dw // block_size
            if min(coarse_h, coarse_w) < 2:
                raise ValueError(
                    f"mask block {block_size} is too large for decoder grid {decoder_grid}"
                )
            length = coarse_h * coarse_w
            count = int(length * ratio)
            noise = torch.rand(num_masks, length, device=device)
            indices = noise.argsort(dim=1)[:, :count]
            mask_2d = torch.zeros(num_masks, length, device=device)
            mask_2d.scatter_(1, indices, 1.0)
            mask_2d = F.interpolate(
                mask_2d.reshape(num_masks, 1, coarse_h, coarse_w),
                size=decoder_grid,
                mode="nearest",
            )

            exact_tiling = dh % block_size == 0 and dw % block_size == 0
            split = dh * dw - count * block_size**2 if exact_tiling else None
        else:
            length = dh * dw
            count = int(length * ratio)
            noise = torch.rand(num_masks, length, device=device)
            indices = noise.argsort(dim=1)[:, :count]
            mask_2d = torch.zeros(num_masks, length, device=device)
            mask_2d.scatter_(1, indices, 1.0)
            mask_2d = mask_2d.reshape(num_masks, 1, dh, dw)
            split = length - count

        return mask_2d, split

    def _source_layout(
        self, source_mask: torch.Tensor, batch: int, split: Optional[int]
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
        if self.attention_mode == "dense" or split is None:
            return None
        mask = (
            source_mask.expand(batch, -1, -1)
            if source_mask.shape[0] == 1
            else source_mask
        )
        order = torch.argsort(mask.squeeze(-1), dim=-1, stable=True)
        inverse = torch.argsort(order, dim=-1)
        return order, inverse, split

    def _run_encoder(
        self,
        tokens: torch.Tensor,
        grid_size: GridSize,
        source_mask: Optional[torch.Tensor] = None,
        split: Optional[int] = None,
    ) -> torch.Tensor:
        if self.pos_encoding == "sincos":
            position = _sincos_2d(
                tokens.shape[-1], grid_size, tokens.device, tokens.dtype
            )
            tokens = tokens + position
            rope = None
        else:
            head_dim = self.blocks[0].attn.head_dim
            rope = _rope_2d(head_dim, grid_size, tokens.device, tokens.dtype)
        tokens = self.pos_drop(tokens)
        layout = (
            None
            if source_mask is None
            else self._source_layout(source_mask, tokens.shape[0], split)
        )

        for block in self.blocks:
            if self.use_checkpoint and self.training:

                def run(value, module=block):
                    return module(value, source_mask, layout, rope)

                tokens = checkpoint(run, tokens, use_reentrant=False)
            else:
                tokens = block(tokens, source_mask, layout, rope)
        return self.norm(tokens)

    def forward_features(self, images: torch.Tensor, return_tokens: bool = False):

        grid_size = self._validate_images(images, paired=False)
        tokens, patch_grid = self.patch_embed(images)
        if patch_grid != grid_size:
            raise RuntimeError("patch embedding returned an unexpected grid")
        tokens = self._run_encoder(tokens, grid_size)
        if return_tokens:
            return tokens, grid_size
        return tokens.mean(dim=1)

    def _pretrain_predictions(self, images: torch.Tensor, mask_ratio: Optional[float]):
        self._validate_images(images, paired=True)
        decoder_mask, encoder_mask, split, encoder_grid, decoder_grid = (
            self._sample_mask(images, mask_ratio)
        )
        tokens, _ = self.patch_embed(images)

        mix_mask = encoder_mask.to(dtype=tokens.dtype)
        tokens = tokens * (1.0 - mix_mask) + tokens.flip(0) * mix_mask
        tokens = self._run_encoder(tokens, encoder_grid, encoder_mask, split)
        if self.decoder_pool is not None:
            tokens = self.decoder_pool(tokens, encoder_grid)

        tokens = self.decoder_embed(tokens)
        batch, length, _ = tokens.shape
        branch_mask = decoder_mask.to(dtype=tokens.dtype)
        mask_token = self.mask_token.to(dtype=tokens.dtype).expand(batch, length, -1)
        first = tokens * (1.0 - branch_mask) + mask_token * branch_mask
        second = tokens * branch_mask + mask_token * (1.0 - branch_mask)
        decoded = torch.cat((first, second), dim=0)
        decoded = decoded + _sincos_2d(
            decoded.shape[-1], decoder_grid, decoded.device, decoded.dtype
        )
        for block in self.decoder_blocks:
            decoded = block(decoded)
        decoded = self.decoder_norm(decoded)
        pred_rgb = self.decoder_pred(decoded)
        pred_diff = self.decoder_pred_diff(decoded)
        return pred_rgb, pred_diff, decoder_mask, encoder_grid, decoder_grid

    @staticmethod
    def _patchify(
        images: torch.Tensor, patch_size: int
    ) -> Tuple[torch.Tensor, GridSize]:
        batch, channels, height, width = images.shape
        h, w = height // patch_size, width // patch_size
        x = images.reshape(batch, channels, h, patch_size, w, patch_size)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(batch, h * w, patch_size * patch_size * channels), (h, w)

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        return self._patchify(images, self.reconstruction_patch_size)[0]

    @staticmethod
    def _unpatchify(
        patches: torch.Tensor, patch_size: int, grid_size: GridSize
    ) -> torch.Tensor:
        batch, length, channels = patches.shape
        h, w = grid_size
        image_channels = channels // (patch_size * patch_size)
        if length != h * w:
            raise ValueError(f"patch count {length} does not match grid {grid_size}")
        x = patches.reshape(batch, h, w, patch_size, patch_size, image_channels)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(batch, image_channels, h * patch_size, w * patch_size)

    def unpatchify(
        self, patches: torch.Tensor, grid_size: Optional[GridSize] = None
    ) -> torch.Tensor:
        if grid_size is None:
            side = math.isqrt(patches.shape[1])
            if side * side != patches.shape[1]:
                raise ValueError("grid_size is required for non-square patch sequences")
            grid_size = (side, side)
        return self._unpatchify(patches, self.reconstruction_patch_size, grid_size)

    def _patchify_channels(self, images: torch.Tensor) -> torch.Tensor:
        return self._patchify(images, self.reconstruction_patch_size)[0]

    def _unmix(self, predictions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch = predictions.shape[0] // 2
        first, second = predictions[:batch], predictions[batch:]
        return first * mask + second.flip(0) * (1.0 - mask)

    def _losses(
        self,
        images,
        pred_rgb,
        pred_diff,
        mask,
        epoch,
        edds_active: Optional[bool] = None,
    ):
        batch = images.shape[0]
        patch_size = self.reconstruction_patch_size
        target_rgb, grid_size = self._patchify(images, patch_size)
        length = target_rgb.shape[1]
        if self.norm_pix_loss:
            p_mean = target_rgb.mean(dim=-1, keepdim=True).detach()
            p_var = target_rgb.var(dim=-1, keepdim=True).detach()
            p_std = (p_var + 1e-6).sqrt().clamp(min=1e-4)
            target_norm = (target_rgb - p_mean) / p_std
        else:
            target_norm, p_mean, p_std = target_rgb, None, None

        unmix_rgb = self._unmix(pred_rgb, mask)
        loss_rgb = (unmix_rgb - target_norm).square().mean()

        with torch.no_grad():
            gx, gy = self.sobel(images)
            magnitude = (gx.square() + gy.square() + 1e-8).sqrt()
            flat = magnitude.flatten(2)
            kth = max(1, int(flat.shape[-1] * self.sobel_q))
            threshold = torch.kthvalue(flat, kth, dim=2).values[:, :, None, None]
            valid = (magnitude > threshold).float()
            gt_direction = F.normalize(torch.stack((gx, gy), dim=-1), dim=-1)
            height, width = images.shape[-2:]
            gt_direction = gt_direction.permute(0, 1, 4, 2, 3).reshape(
                batch, 6, height, width
            )
            gt_direction = self._patchify_channels(gt_direction)
            gt_direction = gt_direction.reshape(batch, length, patch_size**2, 3, 2)
            valid = self._patchify_channels(valid).reshape(
                batch, length, patch_size**2, 3
            )
            num_valid = valid.sum().clamp_min(1.0)

        unmix_diff = self._unmix(pred_diff, mask)
        direction = unmix_diff.reshape(batch, length, patch_size**2, 3, 2)
        direction = (
            direction / (direction.norm(dim=-1, keepdim=True).square() + 1e-4).sqrt()
        )
        cosine = (direction * gt_direction).sum(dim=-1)
        loss_diff = ((1.0 - cosine) * valid).sum() / num_valid

        active = (
            epoch >= self.edds_warmup_epochs
            if edds_active is None
            else bool(edds_active)
        ) and self.lambda_edds > 0
        if not active:
            loss_edds = images.new_zeros(())
        elif self.edds_version == "v2":
            loss_edds = _edds_v2(
                target_rgb,
                unmix_rgb,
                mask,
                self.norm_pix_loss,
                p_std,
                p_mean,
                self.sobel_q,
            )
        else:
            loss_edds = _edds_v1(
                target_rgb,
                unmix_rgb,
                self.norm_pix_loss,
                p_std,
                p_mean,
                self.sobel_q,
            )

        loss = loss_rgb + self.lambda_diff * loss_diff + self.lambda_edds * loss_edds
        return loss, loss_rgb, loss_diff, loss_edds, unmix_rgb, p_mean, p_std, grid_size

    def forward(
        self,
        images: torch.Tensor,
        mask_ratio: Optional[float] = None,
        epoch: int = 0,
        edds_active: Optional[bool] = None,
    ) -> Dict[str, object]:
        pred_rgb, pred_diff, mask, encoder_grid, decoder_grid = (
            self._pretrain_predictions(images, mask_ratio)
        )
        loss, loss_rgb, loss_diff, loss_edds, unmix_rgb, _, _, _ = self._losses(
            images, pred_rgb, pred_diff, mask, epoch, edds_active
        )
        return {
            "loss": loss,
            "pred_rgb": pred_rgb,
            "pred_diff": pred_diff,
            "unmixed_rgb": unmix_rgb,
            "mask": mask,
            "loss_rgb": loss_rgb,
            "loss_diff": loss_diff,
            "loss_edds": loss_edds,
            "encoder_grid": encoder_grid,
            "decoder_grid": decoder_grid,
        }

    @torch.no_grad()
    def reconstruct(
        self, images: torch.Tensor, mask_ratio: Optional[float] = None
    ) -> Dict[str, object]:
        pred_rgb, _, mask, encoder_grid, decoder_grid = self._pretrain_predictions(
            images, mask_ratio
        )
        target, _ = self._patchify(images, self.reconstruction_patch_size)
        unmix_rgb = self._unmix(pred_rgb, mask)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            std = (target.var(dim=-1, keepdim=True) + 1e-6).sqrt().clamp(min=1e-4)
            reconstruction_patches = unmix_rgb * std + mean
        else:
            reconstruction_patches = unmix_rgb
        mixed_patches = target * (1.0 - mask) + target.flip(0) * mask
        reconstruction = self._unpatchify(
            reconstruction_patches, self.reconstruction_patch_size, decoder_grid
        )
        mixed = self._unpatchify(
            mixed_patches, self.reconstruction_patch_size, decoder_grid
        )
        mask_image = self._unpatchify(
            mask.expand(images.shape[0], -1, self.reconstruction_patch_size**2 * 3),
            self.reconstruction_patch_size,
            decoder_grid,
        )
        return {
            "reconstruction": reconstruction,
            "mixed": mixed,
            "mask": mask,
            "mask_image": mask_image,
            "pred_rgb": pred_rgb,
            "unmixed_rgb": unmix_rgb,
            "encoder_grid": encoder_grid,
            "decoder_grid": decoder_grid,
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"mask_token"}

    def get_num_layers(self) -> int:
        return self.depth


MODEL_CONFIGS = {
    "vit_small_patch16": dict(
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_dim=384,
        decoder_depth=6,
        decoder_num_heads=8,
    ),
    "vit_base_patch16": dict(
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
    ),
    "vit_large_patch16": dict(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
    ),
}


_MODEL_ALIASES = {
    "vit-s/16": "vit_small_patch16",
    "vit_s_16": "vit_small_patch16",
    "vit_s16": "vit_small_patch16",
    "vit_small": "vit_small_patch16",
    "vit-b/16": "vit_base_patch16",
    "vit_b_16": "vit_base_patch16",
    "vit_b16": "vit_base_patch16",
    "vit_base": "vit_base_patch16",
    "vit-l/16": "vit_large_patch16",
    "vit_l_16": "vit_large_patch16",
    "vit_l16": "vit_large_patch16",
    "vit_large": "vit_large_patch16",
}


def build_model(name: str = "vit_small_patch16", **kwargs) -> DIMEViT:
    key = _MODEL_ALIASES.get(name.lower(), name.lower())
    if key not in MODEL_CONFIGS:
        choices = ", ".join(MODEL_CONFIGS)
        raise ValueError(f"unknown model {name!r}; choose one of: {choices}")
    config = dict(MODEL_CONFIGS[key])
    config.update(kwargs)
    return DIMEViT(**config)


def vit_small_patch16(**kwargs) -> DIMEViT:
    return build_model("vit_small_patch16", **kwargs)


def vit_base_patch16(**kwargs) -> DIMEViT:
    return build_model("vit_base_patch16", **kwargs)


def vit_large_patch16(**kwargs) -> DIMEViT:
    return build_model("vit_large_patch16", **kwargs)


__all__ = [
    "DIMEViT",
    "MODEL_CONFIGS",
    "build_model",
    "vit_small_patch16",
    "vit_base_patch16",
    "vit_large_patch16",
]
