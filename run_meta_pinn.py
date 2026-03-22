"""
run_meta_pinn.py — The Novel Experiment

Trains MetaWDWNet on (a, p) → Ψ(a; p) for all operator orderings
p ∈ [-2, 4] simultaneously, then generates:

  1. Phase diagram: |Ψ(a, p)| heatmap — Paper Figure 1
  2. Amplitude profiles per p vs. WKB analytical prediction
  3. Tunneling probability ratio as a function of p
  4. Observable summary table

This is the first continuous map of how the wave function of the
universe changes with operator ordering choice.
"""
import sys
sys.path.insert(0, 'C:/Users/point/projects/TryingToMakeBreakThrough_Experimenting')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from src.model import MetaWDWNet
from src.meta_train import train_meta_wdw
from src.physics import wkb_amplitude, solve_wdw_reference


# ── Config ────────────────────────────────────────────────────────────────────
P_MIN      = -2.0
P_MAX      =  4.0
A_MIN      =  0.01
A_MAX      =  3.5
A_REF      =  0.1
N_A_GRID   =  300
N_P_GRID   =  60        # p values for phase diagram
RESULTS    = "results"
os.makedirs(RESULTS, exist_ok=True)

# Physically proposed p values for overlay
P_LANDMARKS = {
    "p=0 (simple)": 0.0,
    "p=1 (Laplace-Beltrami)": 1.0,
    "p=3 (Misner)": 3.0,
}


# ── Train ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  WDW Meta-PINN: Operator Ordering Phase Diagram")
print("=" * 60)
print(f"  Training MetaWDWNet over p in [{P_MIN}, {P_MAX}]")
print(f"  Scale factor a in [{A_MIN}, {A_MAX}]")
print()

model = MetaWDWNet(hidden_dim=256, n_layers=6, omega_0=5.0)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {n_params:,}")
print()

losses = train_meta_wdw(
    model,
    p_min=P_MIN,
    p_max=P_MAX,
    a_min=A_MIN,
    a_max_final=A_MAX,
    n_colloc=4000,
    n_epochs_adam=8000,
    n_epochs_lbfgs=500,
    verbose=True,
)

print("\n  Training complete.")
torch.save(model.state_dict(), f"{RESULTS}/meta_wdw_model.pt")
print(f"  Model saved: {RESULTS}/meta_wdw_model.pt")


# ── Extract observables ───────────────────────────────────────────────────────
a_grid = np.linspace(A_MIN, A_MAX, N_A_GRID)
p_grid = np.linspace(P_MIN, P_MAX, N_P_GRID)

a_tensor = torch.tensor(a_grid, dtype=torch.float32).unsqueeze(1)

# Phase diagram data: |Ψ(a, p)|
print("\n  Extracting phase diagram...")
amplitude_map = np.zeros((N_P_GRID, N_A_GRID))
tunneling_ratios = {}

for i, p_val in enumerate(p_grid):
    p_tensor = torch.full((N_A_GRID, 1), p_val)
    with torch.no_grad():
        psi = model(a_tensor, p_tensor).numpy()
    amp = np.sqrt(psi[:, 0]**2 + psi[:, 1]**2)
    amp_norm = amp / (amp.max() + 1e-12)
    amplitude_map[i] = amp_norm

    # Tunneling ratio: mean amplitude in forbidden region (a<1) / allowed region (a>1)
    forbidden  = amp[a_grid < 1.0].mean()
    allowed    = amp[a_grid > 1.0].mean()
    tunneling_ratios[p_val] = forbidden / (allowed + 1e-12)


# ── Figure 1: Phase Diagram ───────────────────────────────────────────────────
print("  Plotting phase diagram...")
fig, ax = plt.subplots(figsize=(10, 6))

im = ax.imshow(
    amplitude_map,
    extent=[A_MIN, A_MAX, P_MIN, P_MAX],
    origin="lower",
    aspect="auto",
    cmap="inferno",
    interpolation="bilinear",
)
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("|Ψ(a; p)| normalized", fontsize=12)

# Classical turning point
ax.axvline(1.0, color="cyan", linewidth=1.5, linestyle="--", label="Turning point a=1")

# Landmark p values
colors = ["#00ff88", "#ff6b6b", "#4ecdc4"]
for (label, pv), color in zip(P_LANDMARKS.items(), colors):
    ax.axhline(pv, color=color, linewidth=1.2, linestyle=":", alpha=0.9, label=label)

ax.set_xlabel("Scale factor  a", fontsize=13)
ax.set_ylabel("Operator ordering parameter  p", fontsize=13)
ax.set_title(
    "Phase Diagram: Wave Function of the Universe vs. Operator Ordering\n"
    r"$\Psi''(a) + \frac{p}{a}\Psi'(a) - (a^2 - a^4)\Psi(a) = 0$",
    fontsize=13,
)
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
path = f"{RESULTS}/phase_diagram.png"
plt.savefig(path, dpi=180)
plt.close()
print(f"  Saved: {path}")


# ── Figure 2: Amplitude profiles vs. WKB ─────────────────────────────────────
print("  Plotting amplitude profiles vs WKB...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

p_plot_vals = [0.0, 1.0, 3.0]
a_allowed = a_grid[a_grid > 1.05]

for ax, p_val in zip(axes, p_plot_vals):
    p_tensor = torch.full((N_A_GRID, 1), p_val)
    with torch.no_grad():
        psi = model(a_tensor, p_tensor).numpy()
    amp = np.sqrt(psi[:, 0]**2 + psi[:, 1]**2)
    amp_norm = amp / (amp.max() + 1e-12)

    # WKB reference (classically allowed region only)
    wkb_amp = wkb_amplitude(a_allowed, p=p_val)

    # Scipy reference for comparison
    a_ref_arr, psi_ref = solve_wdw_reference(p=p_val, a_min=A_MIN, a_max=A_MAX, n_points=N_A_GRID)
    psi_ref_norm = np.abs(psi_ref) / (np.abs(psi_ref).max() + 1e-12)

    ax.plot(a_grid, amp_norm, label="Meta-PINN |Ψ|", color="#ff6b35", lw=2)
    ax.plot(a_ref_arr, psi_ref_norm, label="scipy ref", color="#4cc9f0",
            lw=1.5, linestyle="--", alpha=0.8)
    ax.plot(a_allowed, wkb_amp / (wkb_amp.max() + 1e-12),
            label="WKB amplitude", color="#7bed9f", lw=1.5, linestyle=":")
    ax.axvline(1.0, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_title(f"p = {p_val}", fontsize=12)
    ax.set_xlabel("Scale factor a", fontsize=11)
    ax.set_ylabel("|Ψ| normalized", fontsize=11)
    ax.legend(fontsize=8)

plt.suptitle("Meta-PINN Amplitude vs. Scipy Reference vs. WKB Prediction", fontsize=13)
plt.tight_layout()
path = f"{RESULTS}/amplitude_profiles.png"
plt.savefig(path, dpi=180)
plt.close()
print(f"  Saved: {path}")


# ── Figure 3: Tunneling ratio vs. p ──────────────────────────────────────────
print("  Plotting tunneling ratio...")
fig, ax = plt.subplots(figsize=(8, 4))

p_vals_arr = np.array(list(tunneling_ratios.keys()))
ratios_arr = np.array(list(tunneling_ratios.values()))

ax.plot(p_vals_arr, ratios_arr, color="#e84393", lw=2)
ax.fill_between(p_vals_arr, ratios_arr, alpha=0.15, color="#e84393")

for label, pv in P_LANDMARKS.items():
    idx = np.argmin(np.abs(p_vals_arr - pv))
    ax.axvline(pv, color="gray", lw=1, linestyle="--", alpha=0.6)
    ax.annotate(label, xy=(pv, ratios_arr[idx]),
                xytext=(pv + 0.1, ratios_arr[idx] * 1.05),
                fontsize=8, color="gray")

ax.set_xlabel("Operator ordering parameter  p", fontsize=13)
ax.set_ylabel("Tunneling ratio  |Ψ|_{a<1} / |Ψ|_{a>1}", fontsize=12)
ax.set_title("Tunneling Probability Proxy vs. Operator Ordering", fontsize=13)
plt.tight_layout()
path = f"{RESULTS}/tunneling_ratio.png"
plt.savefig(path, dpi=180)
plt.close()
print(f"  Saved: {path}")


# ── Training loss curve ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 3))
ax.semilogy(losses, color="#7b2d8b", lw=1.2, alpha=0.8)
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Total loss", fontsize=12)
ax.set_title("Meta-PINN Training Loss", fontsize=12)
plt.tight_layout()
path = f"{RESULTS}/meta_training_loss.png"
plt.savefig(path, dpi=150)
plt.close()
print(f"  Saved: {path}")


# ── Observable Summary Table ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  OBSERVABLE SUMMARY: Tunneling Ratio vs. Operator Ordering")
print("=" * 60)
print(f"  {'p':>8}  {'Tunneling Ratio':>18}  {'Note'}")
print(f"  {'-'*8}  {'-'*18}  {'-'*25}")

for p_val in np.arange(P_MIN, P_MAX + 0.5, 0.5):
    idx = np.argmin(np.abs(p_vals_arr - p_val))
    ratio = ratios_arr[idx]
    note = ""
    if abs(p_val - 0.0) < 0.01: note = "← simple ordering"
    if abs(p_val - 1.0) < 0.01: note = "← Laplace-Beltrami"
    if abs(p_val - 3.0) < 0.01: note = "← Misner ordering"
    print(f"  {p_val:>8.1f}  {ratio:>18.6f}  {note}")

print("=" * 60)
print("\n  All figures saved to results/")
print("  Done.")
