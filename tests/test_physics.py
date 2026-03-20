# tests/test_physics.py
import numpy as np
import pytest
from src.physics import wdw_potential, wkb_amplitude, solve_wdw_reference

def test_potential_sign_change():
    """U(a) must change sign at a=1 (classical turning point)."""
    U_below = wdw_potential(np.array([0.5]))
    U_above = wdw_potential(np.array([1.5]))
    assert U_below > 0, "Potential should be positive (forbidden) for a < 1"
    assert U_above < 0, "Potential should be negative (allowed) for a > 1"

def test_potential_at_turning_point():
    """U(1.0) = 1 - 1 = 0."""
    assert abs(wdw_potential(np.array([1.0]))[0]) < 1e-10

def test_wkb_amplitude_shape():
    """WKB amplitude must match input array shape."""
    a = np.linspace(1.1, 3.0, 50)
    amp = wkb_amplitude(a, p=0)
    assert amp.shape == a.shape

def test_wkb_amplitude_positive():
    """WKB amplitude must be positive in classically allowed region."""
    a = np.linspace(1.1, 2.5, 30)
    assert np.all(wkb_amplitude(a, p=0) > 0)

def test_scipy_reference_returns_array():
    """scipy solver must return a Ψ array over the domain."""
    a_grid, psi = solve_wdw_reference(p=0, a_min=0.01, a_max=3.0, n_points=100)
    assert len(a_grid) == 100
    assert len(psi) == 100
    assert np.all(np.isfinite(psi))
