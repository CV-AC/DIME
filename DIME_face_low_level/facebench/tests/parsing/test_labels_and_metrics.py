from pathlib import Path

import cv2
import numpy as np

from facebench.tasks.parsing.data import compose_celeb_label
from facebench.tasks.parsing.labels import CELEBAMASK_HQ_LABELS, LAPA_LABELS
from facebench.tasks.parsing.metrics import ConfusionMatrix


def test_label_spaces_follow_farl_order():
    assert LAPA_LABELS.num_classes == 11
    assert CELEBAMASK_HQ_LABELS.num_classes == 19
    assert CELEBAMASK_HQ_LABELS.suffixes[:3] == (
        "neck",
        "skin",
        "cloth",
    )
    assert CELEBAMASK_HQ_LABELS.names[15] == "glasses"


def test_celeb_component_overlap_uses_later_class(tmp_path: Path):
    annotation = tmp_path / "CelebAMask-HQ-mask-anno" / "0"
    annotation.mkdir(parents=True)
    neck = np.zeros((512, 512), dtype=np.uint8)
    skin = np.zeros_like(neck)
    neck[3:7, 3:7] = 255
    skin[5:9, 5:9] = 255
    assert cv2.imwrite(str(annotation / "00000_neck.png"), neck)
    assert cv2.imwrite(str(annotation / "00000_skin.png"), skin)
    label = compose_celeb_label(tmp_path, 0)
    assert label[4, 4] == 1
    assert label[6, 6] == 2
    assert label[8, 8] == 2


def test_confusion_matrix_metrics():
    target = np.asarray([[0, 1], [1, 2]])
    prediction = np.asarray([[0, 1], [2, 2]])
    matrix = ConfusionMatrix(3)
    matrix.update(target, prediction)
    values = matrix.summarize(("background", "a", "b"))
    assert values["pixel_accuracy"] == 0.75
    assert values["per_class"]["background"]["f1"] == 1.0
    np.testing.assert_allclose(values["per_class"]["a"]["f1"], 2.0 / 3.0)
    np.testing.assert_allclose(values["per_class"]["b"]["f1"], 2.0 / 3.0)
