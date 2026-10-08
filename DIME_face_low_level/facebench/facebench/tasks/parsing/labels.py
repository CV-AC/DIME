from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelSpace:
    dataset: str
    names: tuple[str, ...]
    suffixes: tuple[str, ...] = ()

    @property
    def num_classes(self) -> int:
        return len(self.names)


LAPA_LABELS = LabelSpace(
    dataset="lapa",
    names=(
        "background",
        "face",
        "left_brow",
        "right_brow",
        "left_eye",
        "right_eye",
        "nose",
        "upper_lip",
        "inner_mouth",
        "lower_lip",
        "hair",
    ),
)


CELEBAMASK_HQ_LABELS = LabelSpace(
    dataset="celebamask_hq",
    names=(
        "background",
        "neck",
        "face",
        "cloth",
        "left_ear",
        "right_ear",
        "left_brow",
        "right_brow",
        "left_eye",
        "right_eye",
        "nose",
        "inner_mouth",
        "lower_lip",
        "upper_lip",
        "hair",
        "glasses",
        "hat",
        "earring",
        "necklace",
    ),
    suffixes=(
        "neck",
        "skin",
        "cloth",
        "l_ear",
        "r_ear",
        "l_brow",
        "r_brow",
        "l_eye",
        "r_eye",
        "nose",
        "mouth",
        "l_lip",
        "u_lip",
        "hair",
        "eye_g",
        "hat",
        "ear_r",
        "neck_l",
    ),
)


LABEL_SPACES = {
    LAPA_LABELS.dataset: LAPA_LABELS,
    CELEBAMASK_HQ_LABELS.dataset: CELEBAMASK_HQ_LABELS,
}


def label_space(dataset: str) -> LabelSpace:
    key = dataset.strip().lower()
    try:
        return LABEL_SPACES[key]
    except KeyError as error:
        raise ValueError(f"Unsupported face-parsing dataset {dataset!r}") from error
