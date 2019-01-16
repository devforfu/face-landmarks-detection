import torch
from torch.nn import functional as F


def rmse_loss(input, target, **kwargs):
    return torch.sqrt(F.mse_loss(input, target, **kwargs))
