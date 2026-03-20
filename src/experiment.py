# src/experiment.py
"""
Experiment runner for the WDW operator ordering study.
Phase 1: Validate PINN against scipy reference solver.
Phase 2: Meta-PINN training (added in Task 7).
Phase 3+: Observable extraction and phase diagram (added in Task 8).
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import os

from src.model import WDWNet
from src.physics import solve_wdw_reference
from src.train import train_wdw


def validate_pinn_vs_scipy(p: float = 0.0, save_plot: bool = True) -> float:
    """
    Train WDWNet for a given p and compare against scipy DOP853 reference.
    Returns L2 relative error.
    """
    print(f"\n=== Validating PINN for p={p} ===")

    model = WDWNet(hidden_dim=128, n_layers=4, omega_0=30.0)
    losses = train_wdw(
        model, p=p,
        n_epochs_adam=3000,
        n_epochs_lbfgs=200,
        verbose=True,
    )

    # Evaluate PINN
    a_grid = torch.linspace(0.01, 3.0, 300).unsqueeze(1)
    with torch.no_grad():
        psi_pinn = model(a_grid).numpy()
    psi_r_pinn = psi_pinn[:, 0]

    # Reference solution
    a_ref, psi_ref = solve_wdw_reference(p=p, a_min=0.01, a_max=3.0, n_points=300)

    # Normalize both for comparison (sign/scale can differ)
    psi_r_norm = psi_r_pinn / (np.abs(psi_r_pinn).max() + 1e-12)
    psi_ref_norm = psi_ref / (np.abs(psi_ref).max() + 1e-12)

    l2_error = np.sqrt(np.mean((psi_r_norm - psi_ref_norm)**2))
    print(f"L2 relative error for p={p}: {l2_error:.4f}")

    if save_plot:
        os.makedirs("results", exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        a_np = a_grid.numpy().squeeze()
        ax1.plot(a_ref, psi_ref_norm, label="scipy DOP853", lw=2)
        ax1.plot(a_np, psi_r_norm, "--", label="PINN Re(Ψ)", lw=2)
        ax1.axvline(1.0, color="gray", linestyle=":", label="turning point a=1")
        ax1.set_xlabel("Scale factor a")
        ax1.set_ylabel("Ψ (normalized)")
        ax1.set_title(f"PINN vs. scipy reference (p={p})")
        ax1.legend()

        ax2.semilogy(losses)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Total loss")
        ax2.set_title("Training loss curve")

        plt.tight_layout()
        plt.savefig(f"results/validation_p{p}.png", dpi=150)
        plt.close()
        print(f"Plot saved: results/validation_p{p}.png")

    return l2_error


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WDW PINN Experiment")
    parser.add_argument("--mode", choices=["validate", "meta", "all"], default="validate")
    args = parser.parse_args()

    if args.mode in ("validate", "all"):
        print("\n=== Phase 1: Validation ===")
        errors = {}
        for p_val in [0.0, 1.0, 2.0]:
            errors[p_val] = validate_pinn_vs_scipy(p=p_val, save_plot=True)

        print("\n=== Validation Results ===")
        for p_val, err in errors.items():
            status = "PASS" if err < 0.10 else "NEEDS_MORE_TRAINING"
            print(f"  p={p_val}: L2 error = {err:.4f}  [{status}]")
