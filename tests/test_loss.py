# tests/test_loss.py
import torch
import pytest
from src.loss import pde_residual, bc_loss_hartle_hawking, normalization_loss, total_loss


def test_pde_residual_near_zero_for_known_solution():
    """
    For p=0 and constant Ψ=1, the PDE residual is U(a)·1 = a²-a⁴.
    At a=1 this is 0, so residual must be near zero there.
    """
    a = torch.tensor([[1.0]], requires_grad=True)
    psi_r = torch.ones(1, 1)
    psi_i = torch.zeros(1, 1)
    res = pde_residual(a, psi_r, psi_i, p=0.0)
    assert res.item() < 1e-8

def test_pde_residual_nonzero_away_from_turning_point():
    """Away from turning point, constant Ψ=1 does not satisfy WDW."""
    a = torch.tensor([[0.5]], requires_grad=True)
    psi_r = torch.ones(1, 1)
    psi_i = torch.zeros(1, 1)
    res = pde_residual(a, psi_r, psi_i, p=0.0)
    assert res.item() > 0.01

def test_bc_hartle_hawking_zero_for_correct_values():
    """BC loss must be zero if Ψ'(a_min)=0 and Ψ(a_max)=0."""
    psi_at_origin = torch.tensor([[1.0, 0.0]])
    dpsi_at_origin = torch.tensor([[0.0, 0.0]])
    psi_at_amax = torch.tensor([[0.0, 0.0]])
    loss = bc_loss_hartle_hawking(psi_at_origin, dpsi_at_origin, psi_at_amax)
    assert loss.item() < 1e-10

def test_normalization_loss_zero_when_pinned():
    """Normalization loss = 0 when Ψ(a_ref) = [1, 0]."""
    psi_at_ref = torch.tensor([[1.0, 0.0]])
    loss = normalization_loss(psi_at_ref)
    assert loss.item() < 1e-10

def test_normalization_loss_nonzero_when_wrong():
    psi_at_ref = torch.tensor([[0.5, 0.3]])
    loss = normalization_loss(psi_at_ref)
    assert loss.item() > 0.1
