from .base import Flow
from .modules import Affine
import torch
import torch.distributions as td


class MaskedAutoregressiveFLow(Flow):
    def __init__(self, input_dim, hidden_dim, n_layers):
        self.base_distribution = td.Normal(torch.tensor(0.), torch.tensor(1.))
        bijectors = [Affine(input_dim, hidden_dim) for _ in range(n_layers)]
        super(MaskedAutoregressiveFLow, self).__init__(*bijectors)
        return

    def log_prob(self, x, y=None):
        u, acc_log_abs_det_jacobians = self.forward(x, y)
        return torch.sum(self.base_distribution.log_prob(u) + acc_log_abs_det_jacobians, dim=1)
