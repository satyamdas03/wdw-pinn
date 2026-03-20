# src/model.py
import torch
import torch.nn as nn
import numpy as np


class SIRENLayer(nn.Module):
    """
    Single sinusoidal layer from SIREN (Sitzmann et al. 2020).
    Activation: sin(omega_0 * Wx + b)
    """
    def __init__(self, in_features: int, out_features: int,
                 omega_0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                n = self.linear.weight.shape[1]
                self.linear.weight.uniform_(-1.0 / n, 1.0 / n)
            else:
                n = self.linear.weight.shape[1]
                bound = np.sqrt(6.0 / n) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class WDWNet(nn.Module):
    """
    1D PINN for Wheeler-DeWitt equation.
    Input:  a (scale factor), shape [N, 1]
    Output: [Re(Ψ), Im(Ψ)],  shape [N, 2]
    """
    def __init__(self, hidden_dim: int = 256, n_layers: int = 5,
                 omega_0: float = 30.0):
        super().__init__()
        self.first = SIRENLayer(1, hidden_dim, omega_0=omega_0, is_first=True)
        self.hidden = nn.ModuleList([
            SIRENLayer(hidden_dim, hidden_dim, omega_0=omega_0)
            for _ in range(n_layers - 1)
        ])
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        x = self.first(a)
        for layer in self.hidden:
            x = layer(x)
        return self.output(x)


class MetaWDWNet(nn.Module):
    """
    Meta-PINN for Wheeler-DeWitt equation over all operator orderings.
    Input:  (a, p) — scale factor + ordering parameter, shape [N, 2]
    Output: [Re(Ψ), Im(Ψ)],  shape [N, 2]
    """
    def __init__(self, hidden_dim: int = 256, n_layers: int = 6,
                 omega_0: float = 30.0):
        super().__init__()
        self.first = SIRENLayer(2, hidden_dim, omega_0=omega_0, is_first=True)
        self.hidden = nn.ModuleList([
            SIRENLayer(hidden_dim, hidden_dim, omega_0=omega_0)
            for _ in range(n_layers - 1)
        ])
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, a: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        x = torch.cat([a, p], dim=-1)
        x = self.first(x)
        for layer in self.hidden:
            x = layer(x)
        return self.output(x)
