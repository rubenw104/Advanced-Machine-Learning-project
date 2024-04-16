from dataclasses import dataclass, field
from typing import Optional
import torch.nn as nn
import torch


class CrossEntropyLoss(nn.Module):
    @dataclass
    class Parameters:
        ignore_index: int
        weight: Optional[torch.Tensor] = field(default=None)

    def __init__(self, parameters: Parameters):
        super(CrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=parameters.weight, ignore_index=parameters.ignore_index)

    def forward(self, outputs, targets):
        return self.ce.forward(outputs, targets)
