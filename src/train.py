# src/train.py
"""
Training loop for Wheeler-DeWitt PINN.

Curriculum: expand domain from tunneling region → full domain.
Optimizer:  Adam (rapid convergence) → L-BFGS (tight residuals).
RAR:        adaptive collocation sampling every 1000 steps.
"""
from __future__ import annotations
import torch
import numpy as np
from tqdm import tqdm

from src.loss import pde_residual, bc_loss_hartle_hawking, normalization_loss, total_loss


def sample_collocation(a_min: float, a_max: float, n: int) -> torch.Tensor:
    """Uniform random collocation points on [a_min, a_max]."""
    return torch.FloatTensor(n, 1).uniform_(a_min, a_max).requires_grad_(True)


def rar_resample(
    model_fn,
    a_colloc: torch.Tensor,
    p_val: float,
    n_add: int,
    a_min: float,
    a_max: float,
) -> torch.Tensor:
    """
    Residual-Adaptive Refinement: add collocation points where PDE residual is largest.
    Uses a proxy (|U(a) * Ψ|) as a cheap residual estimate.
    """
    a_dense = np.linspace(a_min, a_max, 1000)
    a_tensor = torch.tensor(a_dense, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        psi = model_fn(a_tensor)
        U = a_tensor**2 - a_tensor**4
        proxy = (U * psi[:, 0:1]).abs().squeeze().numpy()
        proxy = proxy + 1e-10  # avoid zero probability
        prob = proxy / proxy.sum()

    chosen = np.random.choice(a_dense, size=n_add, p=prob)
    return torch.tensor(chosen, dtype=torch.float32).unsqueeze(1).requires_grad_(True)


def train_wdw(
    model,
    p: float = 0.0,
    bc_type: str = "hartle_hawking",
    a_ref: float = 0.1,
    a_max_final: float = 3.5,
    n_colloc: int = 3000,
    n_epochs_adam: int = 5000,
    n_epochs_lbfgs: int = 500,
    lambda_bc: float = 100.0,
    lambda_norm: float = 50.0,
    verbose: bool = True,
) -> list[float]:
    """
    Train a WDWNet for a fixed operator ordering p.
    Returns: list of total loss values per epoch.
    """
    losses = []
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Curriculum: (a_max, lambda_pde, n_epochs)
    curriculum = [
        (1.2,          0.1,  n_epochs_adam // 3),
        (2.0,          1.0,  n_epochs_adam // 3),
        (a_max_final,  1.0,  n_epochs_adam - 2 * (n_epochs_adam // 3)),
    ]

    a_min = 0.01

    # --- Adam phase with curriculum ---
    for (a_max_curr, lambda_pde, n_ep) in curriculum:
        a_colloc = sample_collocation(a_min, a_max_curr, n_colloc)
        bar = tqdm(range(n_ep), desc=f"Adam a_max={a_max_curr:.1f}", disable=not verbose)

        for epoch in bar:
            optimizer_adam.zero_grad()
            a_c = a_colloc.detach().requires_grad_(True)

            psi = model(a_c)
            psi_r, psi_i = psi[:, 0:1], psi[:, 1:2]

            a_bc_min = torch.tensor([[a_min]], requires_grad=True)
            a_bc_max = torch.tensor([[a_max_final]], requires_grad=True)
            a_bc_ref = torch.tensor([[a_ref]], requires_grad=True)

            psi_min = model(a_bc_min)
            psi_max = model(a_bc_max)
            psi_ref = model(a_bc_ref)
            dpsi_min = torch.autograd.grad(psi_min.sum(), a_bc_min, create_graph=True)[0]

            L_pde = pde_residual(a_c, psi_r, psi_i, p=p)
            L_bc = bc_loss_hartle_hawking(psi_min, dpsi_min, psi_max)
            L_norm = normalization_loss(psi_ref)
            L = total_loss(L_pde, L_bc, L_norm, lambda_pde, lambda_bc, lambda_norm)

            L.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_adam.step()

            losses.append(L.item())
            if epoch % 200 == 0:
                bar.set_postfix(loss=f"{L.item():.4e}")

            # RAR every 1000 steps
            if epoch % 1000 == 999:
                def _model_fn(a_t):
                    return model(a_t)
                new_pts = rar_resample(_model_fn, a_colloc, p, n_add=500,
                                       a_min=a_min, a_max=a_max_curr)
                a_colloc = torch.cat([a_colloc.detach(), new_pts], dim=0)[:n_colloc]
                a_colloc = a_colloc.detach().requires_grad_(True)

        bar.close()

    # --- L-BFGS fine-tuning phase ---
    a_colloc = sample_collocation(a_min, a_max_final, n_colloc * 2)
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(), lr=0.1, max_iter=50,
        history_size=50, line_search_fn="strong_wolfe"
    )

    bar = tqdm(range(n_epochs_lbfgs), desc="L-BFGS fine-tune", disable=not verbose)

    for epoch in bar:
        def closure():
            optimizer_lbfgs.zero_grad()
            a_c = a_colloc.detach().requires_grad_(True)
            psi = model(a_c)
            psi_r, psi_i = psi[:, 0:1], psi[:, 1:2]

            a_bc_min = torch.tensor([[a_min]], requires_grad=True)
            a_bc_max = torch.tensor([[a_max_final]], requires_grad=True)
            a_bc_ref = torch.tensor([[a_ref]], requires_grad=True)

            psi_min = model(a_bc_min)
            psi_max = model(a_bc_max)
            psi_ref = model(a_bc_ref)
            dpsi_min = torch.autograd.grad(psi_min.sum(), a_bc_min, create_graph=True)[0]

            L_pde = pde_residual(a_c, psi_r, psi_i, p=p)
            L_bc = bc_loss_hartle_hawking(psi_min, dpsi_min, psi_max)
            L_norm = normalization_loss(psi_ref)
            L = total_loss(L_pde, L_bc, L_norm, 1.0, lambda_bc, lambda_norm)
            L.backward()
            return L

        L = optimizer_lbfgs.step(closure)
        losses.append(L.item() if L is not None else float('nan'))
        if epoch % 50 == 0:
            bar.set_postfix(loss=f"{losses[-1]:.4e}")

    bar.close()
    return losses
