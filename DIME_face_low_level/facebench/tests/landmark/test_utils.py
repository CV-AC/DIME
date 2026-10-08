import torch

from facebench.tasks.landmark.utils import setup_runtime


def test_rocm_runtime_disables_exhaustive_convolution_search(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "hip", "6.3.4", raising=False)
    torch.backends.cudnn.benchmark = True

    setup_runtime()

    assert torch.backends.cudnn.benchmark is False
