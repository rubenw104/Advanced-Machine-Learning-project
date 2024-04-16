from dataclasses import dataclass, field
from typing import Optional

import torch.nn as nn
import torch


class JaccardLoss(nn.Module):

    @dataclass
    class Parameters:
        weight: Optional[torch.Tensor] = field(default=None)

    def __init__(self, parameters: Parameters):
        super(JaccardLoss, self).__init__()
        self.weight = parameters.weight

    def forward(self, outputs, targets):
        # If the weight is not None, multiply the outputs by the reshaped weight
        if self.weight is not None:
            outputs *= self.weight.view(1, 19, 1, 1)

        # Reshape the outputs to a 2D tensor with the number of rows equal to the batch size
        outputs = outputs.reshape(outputs.size()[0], -1)

        # Unsqueeze the targets tensor to add an extra dimension, then expand it to match the shape of outputs
        # Finally, reshape it to a 2D tensor with the number of rows equal to the batch size
        targets = (torch
                   .unsqueeze(targets, dim=1)
                   .expand(-1, 19, -1, -1)
                   .reshape(targets.size()[0], -1))

        # Calculate the Jaccard index as the ratio of the sum of element-wise minimums to the sum of element-wise maximums
        # Add a small constant to the denominator to avoid division by zero
        jaccard = ((torch.sum(torch.min(outputs, targets), dim=1, keepdim=True) + 1e-8) /
                   (torch.sum(torch.max(outputs, targets), dim=1, keepdim=True) + 1e-8))

        # Return the complement of the mean Jaccard index (i.e., 1 - mean(Jaccard index))
        return 1 - torch.mean(jaccard)
