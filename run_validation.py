"""
Full PINN validation for Wheeler-DeWitt equation.
Runs p=0, 1, 2 with default epoch counts: 5000 Adam + 500 L-BFGS.
"""
import sys
sys.path.insert(0, 'C:/Users/point/projects/TryingToMakeBreakThrough_Experimenting')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from src.model import WDWNet
from src.physics import solve_wdw_reference
from src.train import train_wdw


def validate_full(p: float, hidden_dim: int = 128, save_plot: bool = True) -> float:
    """
    Train WDWNet for a given p using full default epochs and compare against scipy DOP853.
    Returns L2 relative error (shape agreement, normalized).
    """
    print(f"\n=== Validating PINN for p={p} (hidden_dim={hidden_dim}) ===")

    model = WDWNet(hidden_dim=hidden_dim, n_layers=4, omega_0=5.0)
    losses = train_wdw(
        model, p=p,
        n_epochs_adam=5000,
        n_epochs_lbfgs=500,
        verbose=True,
    )

    # Evaluate PINN
    a_grid = torch.linspace(0.01, 3.0, 300).unsqueeze(1)
    with torch.no_grad():
        psi_pinn = model(a_grid).numpy()
    psi_r_pinn = psi_pinn[:, 0]

    # Reference solution
    a_ref, psi_ref = solve_wdw_reference(p=p, a_min=0.01, a_max=3.0, n_points=300)

    # Normalize both for shape comparison
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
        fname = f"results/validation_p{int(p)}_full.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Plot saved: {fname}")

    return l2_error


if __name__ == "__main__":
    results = {}
    for p in [0.0, 1.0, 2.0]:
        err = validate_full(p=p, hidden_dim=128, save_plot=True)
        results[p] = err
        print(f"p={p}: L2={err:.4f}")

    print("\n=== Summary ===")
    for p, err in results.items():
        status = "PASS" if err < 0.15 else "HIGH"
        print(f"  p={p}: L2={err:.4f}  [{status}]")

    # Re-run any HIGH results with hidden_dim=256
    retry_results = {}
    for p, err in results.items():
        if err > 0.20:
            print(f"\nRe-running p={p} with hidden_dim=256 ...")
            err256 = validate_full(p=p, hidden_dim=256, save_plot=True)
            retry_results[p] = err256
            print(f"  p={p} (256): L2={err256:.4f}  [{'PASS' if err256 < 0.15 else 'HIGH'}]")

    if retry_results:
        print("\n=== Final Results (after retry with hidden_dim=256) ===")
        for p in [0.0, 1.0, 2.0]:
            final_err = retry_results.get(p, results[p])
            status = "PASS" if final_err < 0.15 else "HIGH"
            print(f"  p={p}: L2={final_err:.4f}  [{status}]")
