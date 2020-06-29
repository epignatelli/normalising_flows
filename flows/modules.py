import torch.nn.functional as F
from torch import nn
from .core import Bijector


class Affine(Bijector, nn.Linear):
    """
    Scales and shifts into a standard multivariate Gaussian
    """
    def forward(self, u, y=None):
        x = F.linear(u, self.weight, self.bias)
        if y is not None:
            x += F.linear(y, self.weight, self.bias)
        return x

    def inverse(self, x):
        return

    def inverse_log_det_jacobian(self, x):
        raise NotImplementedError
