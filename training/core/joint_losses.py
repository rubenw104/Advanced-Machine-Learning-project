import torch.nn as nn
import torch

from core.cross_entropy_loss import CrossEntropyLoss
from core.focal_loss import FocalLoss
from core.jaccard_loss import JaccardLoss
from core.logitnorm import LogitNormLoss

def _forward(outputs, targets, l1: nn.Module, l2: nn.Module, alpha=0.5, beta=0.5):
    return alpha * l1.forward(outputs, targets) + beta * l2.forward(outputs, targets)


class Joint_LnFl(nn.Module):
    """
    Joint: Logit Normalization & Focal Loss
    """
    def __init__(self, ln_parameters: LogitNormLoss.Parameters, fl_parameters: FocalLoss.Parameters, alpha: float = 0.5, beta: float = 0.5):
        super(Joint_LnFl, self).__init__()
        self.ln = LogitNormLoss(ln_parameters)
        self.fl = FocalLoss(fl_parameters)
        self.alpha = alpha
        self.beta = beta

    def forward(self, outputs, targets):
        return _forward(outputs, targets, self.ln, self.fl, self.alpha, self.beta)


class Joint_LnCe(nn.Module):
    """
    Joint: Logit Normalization & Cross Entropy
    """
    def __init__(self, ln_parameters: LogitNormLoss.Parameters, ce_parameters: CrossEntropyLoss.Parameters, alpha: float = 0.5, beta: float = 0.5):
        super(Joint_LnCe, self).__init__()
        self.ln = LogitNormLoss(ln_parameters)
        self.ce = CrossEntropyLoss(ce_parameters)
        self.alpha = alpha
        self.beta = beta

    def forward(self, outputs, targets):
        return _forward(outputs, targets, self.ln, self.ce, self.alpha, self.beta)


class Joint_JlFl(nn.Module):
    """
    Joint: Jaccard Loss & Focal Loss
    """
    def __init__(self, jl_parameters: JaccardLoss.Parameters, fl_parameters: FocalLoss.Parameters, alpha: float = 0.5, beta: float = 0.5):
        super(Joint_JlFl, self).__init__()
        self.jl = JaccardLoss(jl_parameters)
        self.fl = FocalLoss(fl_parameters)
        self.alpha = alpha
        self.beta = beta

    def forward(self, outputs, targets):
        return _forward(outputs, targets, self.jl, self.fl, self.alpha, self.beta)


class Joint_JlCe(nn.Module):
    """
    Joint: Jaccard Loss & Cross Entropy
    """
    def __init__(self, jl_parameters: JaccardLoss.Parameters, ce_parameters: CrossEntropyLoss.Parameters, alpha: float = 0.5, beta: float = 0.5):
        super(Joint_JlCe, self).__init__()
        self.jl = JaccardLoss(jl_parameters)
        self.ce = CrossEntropyLoss(ce_parameters)
        self.alpha = alpha
        self.beta = beta

    def forward(self, outputs, targets):
        return _forward(outputs, targets, self.jl, self.ce, self.alpha, self.beta)