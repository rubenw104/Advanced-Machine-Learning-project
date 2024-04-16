from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class FocalLoss(nn.Module):
    @dataclass
    class Parameters:
        ignore_index: int
        alpha: Optional[float] = field(default=0.25)
        gamma: float = field(default=2.0)
        reduction: str = field(default="mean")
        weight: Optional[Tensor] = field(default=None)


    def __init__(
            self, parameter: Parameters
    ) -> None:
        super().__init__()
        self.ignore_index: int = parameter.ignore_index
        self.alpha: Optional[float] = parameter.alpha
        self.gamma: float = parameter.gamma
        self.reduction: str = parameter.reduction
        self.weight: Optional[Tensor] = parameter.weight

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        logpt = F.log_softmax(pred, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index, reduction=self.reduction)
        return loss
