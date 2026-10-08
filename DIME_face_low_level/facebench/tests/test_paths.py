import pytest

from facebench.tasks.head_pose.config import resolve_path as pose_path
from facebench.tasks.landmark.config import resolve_path as landmark_path
from facebench.tasks.parsing.config import resolve_path as parsing_path


@pytest.mark.parametrize("resolve", [pose_path, landmark_path, parsing_path])
def test_dataset_paths_use_configured_data_root(resolve, monkeypatch, tmp_path):
    monkeypatch.setenv("FACEBENCH_DATA_ROOT", str(tmp_path))
    expected = tmp_path / "datasets/example/images.lmdb"
    assert resolve("../datasets/example/images.lmdb") == expected
