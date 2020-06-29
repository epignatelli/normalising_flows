from torch import nn
from abc import ABC, abstractmethod


class Bijector(nn.Module, ABC):
    @abstractmethod
    def forward(self, u):
        raise NotImplementedError

    @abstractmethod
    def inverse(self, x):
        raise NotImplementedError

    @abstractmethod
    def inverse_log_det_jacobian(self, x):
        raise NotImplementedError


class TransformedDistribution(nn.Module, ABC):
    def __init__(self, base_distribution, bijector):
        super(TransformedDistribution, self).__init__()
        self.base_distribution = base_distribution
        self.bijector = bijector
        return

    @abstractmethod
    def forward(self, u):
        raise NotImplementedError


class Flow(nn.Sequential):
    def forward(self, x, y):
        acc_log_abs_det_jacobian = 0
        for bijector in self:
            x, log_abs_det_jacobian = bijector(x, y)
            acc_log_abs_det_jacobian += log_abs_det_jacobian
        return x, acc_log_abs_det_jacobian

    def inverse(self, u, y):
        acc_log_abs_det_jacobian = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            acc_log_abs_det_jacobian += log_abs_det_jacobian
        return u, acc_log_abs_det_jacobian

    def log_prob(self, x, y=None):
        raise NotImplementedError
