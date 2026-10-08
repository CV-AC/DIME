from copy import deepcopy

import torch


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0,1).")
        self.decay = float(decay)
        self.module = deepcopy(model).eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        source_parameters = dict(model.named_parameters())
        for name, parameter in self.module.named_parameters():
            parameter.mul_(self.decay).add_(
                source_parameters[name].detach(), alpha=1.0 - self.decay
            )
        source_buffers = dict(model.named_buffers())
        for name, buffer in self.module.named_buffers():
            buffer.copy_(source_buffers[name])

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.module.state_dict()

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.module.load_state_dict(state, strict=True)
