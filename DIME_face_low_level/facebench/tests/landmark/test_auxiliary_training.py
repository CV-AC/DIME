import random
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.io import savemat

from facebench.tasks.landmark.auxiliary_data import (
    AuxiliaryLandmarkDataset,
    DistributedEpochFractionSampler,
    _landmark_box,
    auxiliary_task_schedule,
    build_300w_lp_index,
    build_lapa_index,
    epoch_fractions,
    inspect_auxiliary_index,
)
from facebench.tasks.landmark.config import experiment_output_dir, load_config
from facebench.tasks.landmark.data import FaRLAugment
from facebench.tasks.landmark.model import LandmarkModel, parsing_losses, task_losses
from facebench.tasks.landmark.config import CONFIG_ROOT


from facebench.tasks.landmark.config import LANDMARK_ROOT as ROOT


class _TinyBackbone(torch.nn.Module):
    pyramid_type = "vit"
    out_channels = (16, 16, 16, 16)
    checkpoint_sha256 = "test"

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Conv2d(3, 16, kernel_size=1)

    @staticmethod
    def normalize(images):
        return images

    def forward(self, images):
        feature = torch.nn.functional.adaptive_avg_pool2d(
            self.projection(images), (4, 4)
        )
        return [feature, feature, feature, feature]


def _auxiliary_options():
    return {
        "enabled": True,
        "datasets": {
            "lapa": {
                "enabled": True,
                "num_landmarks": 106,
                "parsing_enabled": True,
                "num_parsing_classes": 11,
            },
            "300w_lp": {"enabled": True, "num_landmarks": 68},
        },
        "loss_weights": {
            "wflw": 1.0,
            "lapa": 1.0,
            "300w_lp": 1.0,
            "lapa_parsing": 1.0,
            "lapa_parsing_dice": 1.0,
        },
    }


def test_disabled_auxiliary_config_preserves_original_output_path() -> None:
    config = load_config(CONFIG_ROOT / "wflw" / "dime_full.yaml")
    config["auxiliary_training"]["enabled"] = False
    expected = (ROOT / "outputs" / "wflw" / "dime").resolve()
    assert experiment_output_dir(config) == expected

    original = LandmarkModel(_TinyBackbone(), input_size=64, head_channels=16)
    disabled = LandmarkModel(
        _TinyBackbone(),
        input_size=64,
        head_channels=16,
        auxiliary_training={"enabled": False},
    )
    assert original.state_dict().keys() == disabled.state_dict().keys()


def test_disabled_robust_augmentation_is_exactly_the_original_pipeline() -> None:
    image = np.full((80, 100, 3), 127, dtype=np.uint8)
    points = np.asarray([[25.0, 20.0], [75.0, 60.0]], dtype=np.float32)
    box = np.asarray([20.0, 10.0, 80.0, 70.0], dtype=np.float32)

    random.seed(19)
    np.random.seed(19)
    original = FaRLAugment(training=True)(image, points, box)
    random.seed(19)
    np.random.seed(19)
    disabled = FaRLAugment(training=True, options={"enabled": False})(
        image, points, box
    )

    for original_value, disabled_value in zip(original, disabled):
        np.testing.assert_array_equal(original_value, disabled_value)


def test_joint_training_uses_the_exact_farl_wflw_augmentation() -> None:
    config = load_config(CONFIG_ROOT / "wflw" / "dime_full.yaml")
    options = config["augmentation"]
    assert options == {
        "preset": "farl_wflw",
        "shift_sigma": 0.05,
        "rot_sigma": 0.174,
        "scale_sigma": 0.10,
        "scale_mu": 0.80,
        "warp_factor": 0.0,
        "noise_fusion_probability": 0.50,
    }
    image = np.full((80, 100, 3), 127, dtype=np.uint8)
    points = np.asarray([[25.0, 20.0], [75.0, 60.0]], dtype=np.float32)
    box = np.asarray([20.0, 10.0, 80.0, 70.0], dtype=np.float32)
    random.seed(23)
    np.random.seed(23)
    baseline = FaRLAugment(training=True)(image, points, box)
    random.seed(23)
    np.random.seed(23)
    joint = FaRLAugment(training=True, options=options)(image, points, box)
    for baseline_value, joint_value in zip(baseline, joint):
        np.testing.assert_array_equal(baseline_value, joint_value)


def test_paper_recipe_does_not_enable_additional_training_data() -> None:
    config = load_config(CONFIG_ROOT / "wflw" / "dime_full.yaml")
    assert config["auxiliary_training"] == {"enabled": False}


def test_landmark_box_expansion_adds_expected_context() -> None:
    points = np.asarray([[10.0, 20.0], [110.0, 80.0]], dtype=np.float32)
    box = _landmark_box(points, expansion=1.25)
    np.testing.assert_allclose(box, [-2.5, -12.5, 122.5, 112.5])


def test_farl_augmentation_reports_only_artificial_occlusion_as_invalid() -> None:
    image = np.full((80, 100, 3), 127, dtype=np.uint8)
    points = np.asarray([[25.0, 20.0], [75.0, 60.0]], dtype=np.float32)
    box = np.asarray([20.0, 10.0, 80.0, 70.0], dtype=np.float32)
    random.seed(29)
    np.random.seed(29)
    ordinary = FaRLAugment(training=True)(image, points, box)
    random.seed(29)
    np.random.seed(29)
    image_augmented, points_augmented, matrix, valid = FaRLAugment(training=True)(
        image, points, box, return_valid_mask=True
    )
    np.testing.assert_array_equal(ordinary[0], image_augmented)
    np.testing.assert_array_equal(ordinary[1], points_augmented)
    np.testing.assert_array_equal(ordinary[2], matrix)
    assert valid.shape == (512, 512)
    assert valid.dtype == np.bool_
    assert valid.any() and (~valid).any()
    invalid_y, invalid_x = np.where(~valid)
    rectangle_area = (invalid_y.max() - invalid_y.min() + 1) * (
        invalid_x.max() - invalid_x.min() + 1
    )
    assert len(invalid_y) == rectangle_area


def test_parsing_loss_ignores_artificial_occlusion_pixels() -> None:
    logits = torch.randn(1, 11, 8, 8)
    first_target = torch.zeros(1, 8, 8, dtype=torch.long)
    second_target = first_target.clone()
    second_target[:, 2:6, 2:6] = 7
    valid = torch.ones(1, 8, 8, dtype=torch.bool)
    valid[:, 2:6, 2:6] = False
    first = parsing_losses(
        logits, first_target, valid_mask=valid, include_background=True
    )
    second = parsing_losses(
        logits, second_target, valid_mask=valid, include_background=True
    )
    torch.testing.assert_close(first["loss"], second["loss"])


def test_auxiliary_output_path_is_isolated() -> None:
    config = load_config(CONFIG_ROOT / "wflw" / "dime_full.yaml")
    config["auxiliary_training"]["enabled"] = True
    assert experiment_output_dir(config).name == "auxiliary"


def test_epoch_fractions_are_independent_dataset_coverage() -> None:
    fractions = epoch_fractions(
        {
            "epoch_fraction": {
                "wflw": 1.0,
                "lapa": 0.5,
                "300w_lp": 0.25,
            }
        },
        ("lapa", "300w_lp"),
    )
    assert fractions == {"wflw": 1.0, "lapa": 0.5, "300w_lp": 0.25}

    sizes = {"wflw": 80, "lapa": 120, "300w_lp": 160}
    samplers = {
        task: DistributedEpochFractionSampler(
            size,
            fraction=fractions[task],
            num_replicas=8,
            rank=0,
            seed=0,
        )
        for task, size in sizes.items()
    }
    assert samplers["wflw"].selected_samples == 80
    assert samplers["lapa"].selected_samples == 60
    assert samplers["300w_lp"].selected_samples == 40

    counts = {task: (len(sampler) + 7) // 8 for task, sampler in samplers.items()}
    first = auxiliary_task_schedule(counts, seed=0, epoch=4)
    second = auxiliary_task_schedule(counts, seed=0, epoch=4)
    assert first == second
    assert {task: first.count(task) for task in counts} == counts


def test_fraction_sampler_covers_wflw_once_with_only_ddp_padding() -> None:
    shards = []
    for rank in range(8):
        sampler = DistributedEpochFractionSampler(
            7_500,
            fraction=1.0,
            num_replicas=8,
            rank=rank,
            seed=3,
        )
        sampler.set_epoch(9)
        shards.extend(list(sampler))
        assert len(sampler) == 938
    assert len(shards) == 7_504
    assert set(shards) == set(range(7_500))
    assert len(shards) - len(set(shards)) == 4


def test_all_task_heads_receive_a_ddp_visible_gradient() -> None:
    for task, landmarks in (("wflw", 98), ("lapa", 106), ("300w_lp", 68)):
        model = LandmarkModel(
            _TinyBackbone(),
            input_size=64,
            head_channels=16,
            auxiliary_training=_auxiliary_options(),
        ).train()
        outputs = model(torch.randn(2, 3, 64, 64), task=task)
        target = torch.rand(2, landmarks, 2) * 63
        parsing = torch.zeros(2, 64, 64, dtype=torch.long) if task == "lapa" else None
        losses = task_losses(
            outputs,
            target,
            task=task,
            auxiliary_training=_auxiliary_options(),
            parsing_target=parsing,
        )
        losses["loss"].backward()
        assert model.backbone.projection.weight.grad is not None
        assert model.backbone.projection.weight.grad.abs().sum() > 0
        assert all(
            parameter.grad is not None
            for name, parameter in model.named_parameters()
            if "classifier" in name or "auxiliary_heads" in name
        )


def test_auxiliary_indexes_and_datasets_round_trip(tmp_path: Path) -> None:
    lapa = tmp_path / "LaPa"
    for split in ("train", "val", "test"):
        for directory in ("images", "labels", "landmarks"):
            (lapa / split / directory).mkdir(parents=True)
    image = np.full((80, 100, 3), 127, dtype=np.uint8)
    mask = np.zeros((80, 100), dtype=np.uint8)
    points106 = np.stack(
        [np.linspace(20, 80, 106), np.linspace(15, 65, 106)], axis=1
    ).astype(np.float32)
    cv2.imwrite(str(lapa / "train" / "images" / "face.jpg"), image)
    cv2.imwrite(str(lapa / "train" / "labels" / "face.png"), mask)
    lines = ["106", *(f"{x} {y}" for x, y in points106)]
    (lapa / "train" / "landmarks" / "face.txt").write_text("\n".join(lines))
    lapa_index = tmp_path / "lapa.npz"
    metadata = build_lapa_index(lapa, lapa_index)
    assert metadata["samples"] == 1
    assert metadata["index_format_version"] == 2
    assert len(metadata["manifest_sha256"]) == 64
    inspect_auxiliary_index(
        lapa_index,
        expected_task="lapa",
        expected_root=lapa,
        expected_num_landmarks=106,
        require_full_decode=True,
    )
    dataset = AuxiliaryLandmarkDataset(
        task="lapa",
        root=lapa,
        index_file=lapa_index,
        augmentation=None,
        parsing_enabled=True,
    )
    sample = dataset[0]
    assert sample["landmarks_canvas"].shape == (106, 2)
    assert sample["parsing_mask"].shape == (512, 512)
    assert sample["parsing_valid_mask"].shape == (512, 512)
    assert sample["parsing_valid_mask"].dtype == torch.bool

    lp = tmp_path / "300W_LP"
    for directory in (
        "AFW",
        "AFW_Flip",
        "HELEN",
        "HELEN_Flip",
        "IBUG",
        "IBUG_Flip",
        "LFPW",
        "LFPW_Flip",
    ):
        target = lp / directory
        target.mkdir(parents=True)
        cv2.imwrite(str(target / "face.jpg"), image)
        points68 = np.stack(
            [np.linspace(20, 80, 68), np.linspace(15, 65, 68)], axis=0
        ).astype(np.float32)
        savemat(target / "face.mat", {"pt2d": points68})
    lp_index = tmp_path / "lp.npz"
    metadata = build_300w_lp_index(lp, lp_index)
    assert metadata["samples"] == 8
    inspect_auxiliary_index(
        lp_index,
        expected_task="300w_lp",
        expected_root=lp,
        expected_num_landmarks=68,
        require_full_decode=True,
    )
    dataset = AuxiliaryLandmarkDataset(
        task="300w_lp",
        root=lp,
        index_file=lp_index,
        augmentation=None,
    )
    assert dataset[0]["landmarks_canvas"].shape == (68, 2)
