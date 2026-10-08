from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


WFLW_HORIZONTAL_FLIP_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 32),
    (1, 31),
    (2, 30),
    (3, 29),
    (4, 28),
    (5, 27),
    (6, 26),
    (7, 25),
    (8, 24),
    (9, 23),
    (10, 22),
    (11, 21),
    (12, 20),
    (13, 19),
    (14, 18),
    (15, 17),
    (33, 46),
    (34, 45),
    (35, 44),
    (36, 43),
    (37, 42),
    (38, 50),
    (39, 49),
    (40, 48),
    (41, 47),
    (55, 59),
    (56, 58),
    (60, 72),
    (61, 71),
    (62, 70),
    (63, 69),
    (64, 68),
    (65, 75),
    (66, 74),
    (67, 73),
    (76, 82),
    (77, 81),
    (78, 80),
    (83, 87),
    (84, 86),
    (88, 92),
    (89, 91),
    (93, 95),
    (96, 97),
)


IBUG68_HORIZONTAL_FLIP_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 16),
    (1, 15),
    (2, 14),
    (3, 13),
    (4, 12),
    (5, 11),
    (6, 10),
    (7, 9),
    (17, 26),
    (18, 25),
    (19, 24),
    (20, 23),
    (21, 22),
    (31, 35),
    (32, 34),
    (36, 45),
    (37, 44),
    (38, 43),
    (39, 42),
    (40, 47),
    (41, 46),
    (48, 54),
    (49, 53),
    (50, 52),
    (55, 59),
    (56, 58),
    (60, 64),
    (61, 63),
    (65, 67),
)


LAPA106_HORIZONTAL_FLIP_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 32),
    (1, 31),
    (2, 30),
    (3, 29),
    (4, 28),
    (5, 27),
    (6, 26),
    (7, 25),
    (8, 24),
    (9, 23),
    (10, 22),
    (11, 21),
    (12, 20),
    (13, 19),
    (14, 18),
    (15, 17),
    (33, 46),
    (34, 45),
    (35, 44),
    (36, 43),
    (37, 42),
    (38, 50),
    (39, 49),
    (40, 48),
    (41, 47),
    (55, 65),
    (56, 64),
    (57, 63),
    (58, 62),
    (59, 61),
    (66, 79),
    (67, 78),
    (68, 77),
    (69, 76),
    (70, 75),
    (71, 82),
    (72, 81),
    (73, 80),
    (74, 83),
    (84, 90),
    (85, 89),
    (86, 88),
    (91, 95),
    (92, 94),
    (96, 100),
    (97, 99),
    (101, 103),
    (104, 105),
)


def _permutation(num_points: int, pairs: Sequence[tuple[int, int]]) -> tuple[int, ...]:
    permutation = list(range(num_points))
    seen: set[int] = set()
    for left, right in pairs:
        if left == right or left in seen or right in seen:
            raise ValueError(f"Invalid or duplicate flip pair ({left}, {right}).")
        if not 0 <= left < num_points or not 0 <= right < num_points:
            raise ValueError(f"Flip pair ({left}, {right}) is out of range.")
        permutation[left], permutation[right] = right, left
        seen.update((left, right))
    result = tuple(permutation)
    if tuple(result[index] for index in result) != tuple(range(num_points)):
        raise ValueError("A horizontal flip permutation must be an involution.")
    return result


WFLW_HORIZONTAL_FLIP_INDEX = _permutation(98, WFLW_HORIZONTAL_FLIP_PAIRS)
IBUG68_HORIZONTAL_FLIP_INDEX = _permutation(68, IBUG68_HORIZONTAL_FLIP_PAIRS)
LAPA106_HORIZONTAL_FLIP_INDEX = _permutation(106, LAPA106_HORIZONTAL_FLIP_PAIRS)


def horizontal_flip_index(schema: str) -> tuple[int, ...] | None:
    schema = schema.strip().lower()
    if schema == "wflw":
        return WFLW_HORIZONTAL_FLIP_INDEX
    if schema == "300w_lp":
        return IBUG68_HORIZONTAL_FLIP_INDEX
    if schema == "lapa":
        return LAPA106_HORIZONTAL_FLIP_INDEX
    raise ValueError(f"Unknown landmark schema {schema!r}.")


def horizontal_flip_matrix(size: int) -> torch.Tensor:

    return torch.tensor(
        [[-1.0, 0.0, float(size - 1)], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )


def _conditionally_reorder(
    tensor: torch.Tensor,
    flipped: torch.Tensor,
    permutation: Sequence[int],
) -> torch.Tensor:
    index = torch.as_tensor(permutation, device=tensor.device, dtype=torch.long)
    reordered = tensor.index_select(1, index)
    condition = flipped.to(device=tensor.device, dtype=torch.bool)
    condition = condition.view(-1, *([1] * (tensor.ndim - 1)))
    return torch.where(condition, reordered, tensor)


def undo_horizontal_flip_heatmaps(
    heatmaps: torch.Tensor, *, schema: str
) -> torch.Tensor:

    permutation = horizontal_flip_index(schema)
    if permutation is None:
        raise ValueError(f"Horizontal flip is not verified for schema {schema!r}.")
    if heatmaps.shape[1] != len(permutation):
        raise ValueError(
            f"{schema} expects {len(permutation)} heatmaps, got {heatmaps.shape[1]}."
        )
    index = torch.as_tensor(permutation, device=heatmaps.device, dtype=torch.long)
    return torch.flip(heatmaps.index_select(1, index), dims=(-1,))


def undo_wflw_horizontal_flip_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:

    return undo_horizontal_flip_heatmaps(heatmaps, schema="wflw")


def points_to_canonical(
    normalized_points: torch.Tensor,
    view_transforms: torch.Tensor,
    canonical_transforms: torch.Tensor,
    flipped: torch.Tensor,
    *,
    canvas_size: int,
) -> torch.Tensor:

    points = normalized_points.float() * float(canvas_size) - 0.5
    ones = torch.ones((*points.shape[:-1], 1), device=points.device)
    homogeneous = torch.cat((points, ones), dim=-1)
    view_to_canonical = canonical_transforms.float() @ torch.linalg.inv(
        view_transforms.float()
    )
    canonical = torch.bmm(homogeneous, view_to_canonical.transpose(1, 2))[..., :2]
    canonical = _conditionally_reorder(canonical, flipped, WFLW_HORIZONTAL_FLIP_INDEX)
    return (canonical + 0.5) / float(canvas_size)


def heatmaps_to_canonical(
    heatmaps: torch.Tensor,
    view_transforms: torch.Tensor,
    canonical_transforms: torch.Tensor,
    flipped: torch.Tensor,
    *,
    canvas_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:

    if heatmaps.ndim != 4:
        raise ValueError("heatmaps must have shape [B,K,H,W].")
    batch, _, height, width = heatmaps.shape
    if height <= 1 or width <= 1:
        raise ValueError("heatmaps must have non-trivial spatial dimensions.")
    heatmaps = _conditionally_reorder(
        heatmaps.float(), flipped, WFLW_HORIZONTAL_FLIP_INDEX
    )

    dtype, device = heatmaps.dtype, heatmaps.device
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )

    canvas_x = (xx + 0.5) * (float(canvas_size) / width) - 0.5
    canvas_y = (yy + 0.5) * (float(canvas_size) / height) - 0.5
    ones = torch.ones_like(canvas_x)
    canonical_grid = torch.stack((canvas_x, canvas_y, ones), dim=-1)
    canonical_grid = canonical_grid.reshape(1, height * width, 3).expand(batch, -1, -1)

    canonical_to_view = view_transforms.float() @ torch.linalg.inv(
        canonical_transforms.float()
    )
    view_grid = torch.bmm(canonical_grid, canonical_to_view.transpose(1, 2))
    x_norm = 2.0 * (view_grid[..., 0] + 0.5) / float(canvas_size) - 1.0
    y_norm = 2.0 * (view_grid[..., 1] + 0.5) / float(canvas_size) - 1.0
    grid = torch.stack((x_norm, y_norm), dim=-1).reshape(batch, height, width, 2)

    warped = F.grid_sample(
        heatmaps,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    valid = F.grid_sample(
        torch.ones((batch, 1, height, width), device=device, dtype=dtype),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return warped, valid.clamp_(0.0, 1.0)


def affine_consistency_losses(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
    first_transform: torch.Tensor,
    second_transform: torch.Tensor,
    canonical_transform: torch.Tensor,
    first_flipped: torch.Tensor,
    second_flipped: torch.Tensor,
    *,
    canvas_size: int = 512,
) -> dict[str, torch.Tensor]:

    first_probability = torch.sigmoid(first["heatmap_logits"].float())
    second_probability = torch.sigmoid(second["heatmap_logits"].float())
    first_heatmap, first_valid = heatmaps_to_canonical(
        first_probability,
        first_transform,
        canonical_transform,
        first_flipped,
        canvas_size=canvas_size,
    )
    second_heatmap, second_valid = heatmaps_to_canonical(
        second_probability,
        second_transform,
        canonical_transform,
        second_flipped,
        canvas_size=canvas_size,
    )
    overlap = first_valid * second_valid
    pixel_loss = F.smooth_l1_loss(
        first_heatmap, second_heatmap, reduction="none", beta=0.01
    )
    heatmap_denominator = (overlap.sum() * first_heatmap.shape[1]).clamp_min(1.0)
    heatmap = (pixel_loss * overlap).sum() / heatmap_denominator

    first_points = points_to_canonical(
        first["points"],
        first_transform,
        canonical_transform,
        first_flipped,
        canvas_size=canvas_size,
    )
    second_points = points_to_canonical(
        second["points"],
        second_transform,
        canonical_transform,
        second_flipped,
        canvas_size=canvas_size,
    )
    inside = (
        (first_points >= 0.0).all(dim=-1)
        & (first_points <= 1.0).all(dim=-1)
        & (second_points >= 0.0).all(dim=-1)
        & (second_points <= 1.0).all(dim=-1)
    )
    coordinate_values = F.smooth_l1_loss(
        first_points, second_points, reduction="none", beta=1.0 / canvas_size
    ).sum(dim=-1)
    coordinate = (coordinate_values * inside).sum() / inside.sum().clamp_min(1)
    return {"heatmap_consistency": heatmap, "coordinate_consistency": coordinate}
