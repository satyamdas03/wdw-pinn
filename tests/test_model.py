# tests/test_model.py
import torch
import pytest
from src.model import SIRENLayer, WDWNet, MetaWDWNet

def test_siren_layer_output_shape():
    layer = SIRENLayer(in_features=1, out_features=64, omega_0=30.0, is_first=True)
    x = torch.randn(100, 1)
    out = layer(x)
    assert out.shape == (100, 64)

def test_siren_layer_uses_sine():
    """Output values must be in [-1, 1] since sin maps to that range."""
    layer = SIRENLayer(in_features=1, out_features=64, omega_0=30.0, is_first=True)
    x = torch.randn(100, 1)
    out = layer(x)
    assert out.min() >= -1.0 - 1e-5
    assert out.max() <= 1.0 + 1e-5

def test_wdwnet_output_shape():
    net = WDWNet(hidden_dim=64, n_layers=3, omega_0=30.0)
    a = torch.linspace(0.01, 3.5, 50).unsqueeze(1)
    out = net(a)
    assert out.shape == (50, 2), "Output must be [Re(Ψ), Im(Ψ)]"

def test_metawdwnet_output_shape():
    net = MetaWDWNet(hidden_dim=64, n_layers=3, omega_0=30.0)
    a = torch.linspace(0.01, 3.5, 50).unsqueeze(1)
    p = torch.full((50, 1), 1.0)
    out = net(a, p)
    assert out.shape == (50, 2)

def test_metawdwnet_different_p_gives_different_output():
    """Changing p must change the output — not a trivial constant."""
    net = MetaWDWNet(hidden_dim=64, n_layers=3, omega_0=30.0)
    a = torch.linspace(0.5, 2.0, 30).unsqueeze(1)
    p0 = torch.zeros(30, 1)
    p2 = torch.full((30, 1), 2.0)
    out0 = net(a, p0)
    out2 = net(a, p2)
    assert not torch.allclose(out0, out2), "Different p must give different Ψ"
