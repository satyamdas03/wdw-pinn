# src/meta_train.py
"""
Training loop for MetaWDWNet — the novel contribution.

MetaWDWNet takes (a, p) as joint input and learns Ψ(a; p) continuously
across all operator orderings p ∈ [-2, 4] in a single forward pass.

This enables the first continuous map of how the wave function of the
universe changes with operator ordering — the phase diagram |Ψ(a, p)|.
"""
from __future__ import annotations
import torch
import numpy as np
from tqdm import tqdm

from src.loss import total_loss


def _wdw_potential_tensor(a: torch.Tensor) -> torch.Tensor:
    """U(a) = a² - a⁴  (turning point at a=1)."""
    return a**2 - a**4


def pde_residual_meta(
    a: torch.Tensor,
    p_tensor: torch.Tensor,
    psi_r: torch.Tensor,
    psi_i: torch.Tensor,
) -> torch.Tensor:
    """
    WDW PDE residual for MetaWDWNet where p is a tensor (per-point ordering).

    Equation: Ψ'' + (p/a)·Ψ' - U(a)·Ψ = 0

    a, p_tensor: shape [N, 1]
    psi_r, psi_i: shape [N, 1]
    """
    U = _wdw_potential_tensor(a)
    a_safe = a.clamp(min=1e-6)

    def _grad(y, x):
        if y.requires_grad or y.grad_fn is not None:
            g = torch.autograd.grad(
                y.sum(), x, create_graph=True, retain_graph=True, allow_unused=True
            )[0]
            return g if g is not None else torch.zeros_like(x)
        return torch.zeros_like(x)

    dpsi_r = _grad(psi_r, a)
    dpsi_i = _grad(psi_i, a)
    d2psi_r = _grad(dpsi_r, a)
    d2psi_i = _grad(dpsi_i, a)

    res_r = d2psi_r + (p_tensor / a_safe) * dpsi_r - U * psi_r
    res_i = d2psi_i + (p_tensor / a_safe) * dpsi_i - U * psi_i

    return torch.mean(res_r**2 + res_i**2)


def sample_meta_collocation(
    a_min: float,
    a_max: float,
    p_min: float,
    p_max: float,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample (a, p) collocation points jointly.
    Returns (a_tensor, p_tensor), each shape [N, 1] with requires_grad.
    """
    a = torch.FloatTensor(n, 1).uniform_(a_min, a_max).requires_grad_(True)
    p = torch.FloatTensor(n, 1).uniform_(p_min, p_max)
    return a, p


def train_meta_wdw(
    model,
    p_min: float = -2.0,
    p_max: float = 4.0,
    a_min: float = 0.01,
    a_max_final: float = 3.5,
    n_colloc: int = 4000,
    n_epochs_adam: int = 8000,
    n_epochs_lbfgs: int = 500,
    lambda_pde: float = 1.0,
    lambda_bc: float = 100.0,
    lambda_norm: float = 50.0,
    a_ref: float = 0.1,
    verbose: bool = True,
) -> list[float]:
    """
    Train MetaWDWNet over all operator orderings simultaneously.

    The network learns Ψ(a; p) for p ∈ [p_min, p_max] in one model.
    Boundary conditions are enforced for each p in the batch.

    Returns: list of total loss per epoch.
    """
    losses = []
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_adam, T_max=n_epochs_adam, eta_min=1e-5
    )

    # Curriculum: progressively expand a domain
    curriculum = [
        (1.2,          0.1,  n_epochs_adam // 3),
        (2.0,          0.5,  n_epochs_adam // 3),
        (a_max_final,  1.0,  n_epochs_adam - 2 * (n_epochs_adam // 3)),
    ]

    # ── Adam with curriculum ──────────────────────────────────────────────
    for (a_max_curr, lam_pde, n_ep) in curriculum:
        bar = tqdm(range(n_ep), desc=f"Meta-Adam a_max={a_max_curr:.1f}", disable=not verbose)

        for epoch in bar:
            optimizer_adam.zero_grad()

            # Sample joint (a, p) collocation points
            a_c, p_c = sample_meta_collocation(a_min, a_max_curr, p_min, p_max, n_colloc)

            psi = model(a_c, p_c)
            psi_r, psi_i = psi[:, 0:1], psi[:, 1:2]

            # PDE residual — p is per-point tensor
            L_pde = pde_residual_meta(a_c, p_c, psi_r, psi_i)

            # Boundary conditions: sample a range of p values for BC enforcement
            p_bc_vals = torch.FloatTensor(20, 1).uniform_(p_min, p_max)
            a_bc_min = torch.full((20, 1), a_min, requires_grad=True)
            a_bc_ref = torch.full((20, 1), a_ref)

            psi_min = model(a_bc_min, p_bc_vals)
            dpsi_min = torch.autograd.grad(
                psi_min.sum(), a_bc_min, create_graph=True
            )[0]

            # HH BC: Ψ'(a_min; p) = 0 for all p
            L_bc = torch.mean(dpsi_min**2)

            # Normalization pin: Ψ(a_ref; p) = [1, 0] for all p
            psi_ref_out = model(a_bc_ref, p_bc_vals)
            target = torch.tensor([[1.0, 0.0]]).expand_as(psi_ref_out)
            L_norm = torch.mean((psi_ref_out - target)**2)

            L = total_loss(L_pde, L_bc, L_norm, lam_pde, lambda_bc, lambda_norm)

            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_adam.step()
            scheduler.step()

            losses.append(L.item())
            if epoch % 200 == 0:
                bar.set_postfix(loss=f"{L.item():.4e}")

        bar.close()

    # ── L-BFGS fine-tuning ────────────────────────────────────────────────
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=0.1, max_iter=50,
        history_size=50, line_search_fn="strong_wolfe"
    )

    bar = tqdm(range(n_epochs_lbfgs), desc="Meta-LBFGS", disable=not verbose)

    for epoch in bar:
        def closure():
            optimizer_lbfgs.zero_grad()
            a_c, p_c = sample_meta_collocation(a_min, a_max_final, p_min, p_max, n_colloc)

            psi = model(a_c, p_c)
            psi_r, psi_i = psi[:, 0:1], psi[:, 1:2]

            L_pde = pde_residual_meta(a_c, p_c, psi_r, psi_i)

            p_bc_vals = torch.FloatTensor(20, 1).uniform_(p_min, p_max)
            a_bc_min = torch.full((20, 1), a_min, requires_grad=True)
            a_bc_ref = torch.full((20, 1), a_ref)

            psi_min = model(a_bc_min, p_bc_vals)
            dpsi_min = torch.autograd.grad(
                psi_min.sum(), a_bc_min, create_graph=True
            )[0]
            L_bc = torch.mean(dpsi_min**2)

            psi_ref_out = model(a_bc_ref, p_bc_vals)
            target = torch.tensor([[1.0, 0.0]]).expand_as(psi_ref_out)
            L_norm = torch.mean((psi_ref_out - target)**2)

            L = total_loss(L_pde, L_bc, L_norm, 1.0, lambda_bc, lambda_norm)
            L.backward()
            return L

        L = optimizer_lbfgs.step(closure)
        losses.append(L.item() if L is not None else float("nan"))
        if epoch % 50 == 0:
            bar.set_postfix(loss=f"{losses[-1]:.4e}")

    bar.close()
    return losses
