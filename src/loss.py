# src/loss.py
import torch


def _wdw_potential(a: torch.Tensor) -> torch.Tensor:
    """U(a) = a² - a⁴  (turning point at a=1)."""
    return a**2 - a**4


def pde_residual(
    a: torch.Tensor,
    psi_r: torch.Tensor,
    psi_i: torch.Tensor,
    p: float,
) -> torch.Tensor:
    """
    Mean squared WDW residual: Ψ'' + (p/a)·Ψ' - U(a)·Ψ
    Uses autograd — a must have requires_grad=True.
    Note: Implements the WDW equation multiplied by -1 (equivalent form); residual=0 is the same condition.
    """
    U = _wdw_potential(a)
    a_safe = a.clamp(min=1e-6)

    def _grad(y, x):
        """Gradient of y w.r.t. x; returns zeros if y has no grad_fn."""
        if y.requires_grad or y.grad_fn is not None:
            g = torch.autograd.grad(y.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
            return g if g is not None else torch.zeros_like(x)
        return torch.zeros_like(x)

    dpsi_r = _grad(psi_r, a)
    dpsi_i = _grad(psi_i, a)
    d2psi_r = _grad(dpsi_r, a)
    d2psi_i = _grad(dpsi_i, a)

    res_r = d2psi_r + (p / a_safe) * dpsi_r - U * psi_r
    res_i = d2psi_i + (p / a_safe) * dpsi_i - U * psi_i

    return torch.mean(res_r**2 + res_i**2)


def bc_loss_hartle_hawking(
    psi_at_origin: torch.Tensor,
    dpsi_at_origin: torch.Tensor,
    psi_at_amax: torch.Tensor,
) -> torch.Tensor:
    """
    Hartle-Hawking BCs:
      1. Regularity: Ψ'(a_min) = 0
      2. Decay: Ψ(a_max) → 0
    """
    loss_reg = torch.sum(dpsi_at_origin**2)
    loss_decay = torch.sum(psi_at_amax**2)
    return loss_reg + loss_decay


def bc_loss_vilenkin(
    psi_at_amax: torch.Tensor,
    dpsi_at_amax: torch.Tensor,
    a_max: float,
) -> torch.Tensor:
    """
    Vilenkin BC: outgoing wave only at a_max.
    Condition: Ψ' = +i·k·Ψ  (outgoing → expanding universe)
    """
    U_amax = a_max**2 - a_max**4
    k = float((-U_amax) ** 0.5)

    psi_r, psi_i = psi_at_amax[:, 0:1], psi_at_amax[:, 1:2]
    dpsi_r, dpsi_i = dpsi_at_amax[:, 0:1], dpsi_at_amax[:, 1:2]

    res_r = dpsi_r + k * psi_i
    res_i = dpsi_i - k * psi_r

    return torch.mean(res_r**2 + res_i**2)


def normalization_loss(psi_at_ref: torch.Tensor) -> torch.Tensor:
    """Pin Ψ(a_ref) = [1, 0] to prevent trivial Ψ=0 solution."""
    target = torch.tensor([[1.0, 0.0]], dtype=psi_at_ref.dtype, device=psi_at_ref.device)
    return torch.mean((psi_at_ref - target)**2)


def total_loss(
    L_pde: torch.Tensor,
    L_bc: torch.Tensor,
    L_norm: torch.Tensor,
    lambda_pde: float = 1.0,
    lambda_bc: float = 100.0,
    lambda_norm: float = 50.0,
) -> torch.Tensor:
    """Weighted sum. BC dominates early to prevent Ψ=0 collapse."""
    return lambda_pde * L_pde + lambda_bc * L_bc + lambda_norm * L_norm
