<div align="center">

# 🌌 WDW-PINN

### Physics-Informed Neural Networks for the Wheeler-DeWitt Equation

*A computational study of operator ordering ambiguity in quantum cosmology*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-15%20passing-brightgreen?style=flat-square)](#testing)
[![Status](https://img.shields.io/badge/Status-Active%20Research-orange?style=flat-square)](#project-status)

---

**The Wheeler-DeWitt equation is the Schrödinger equation of the universe.**
Nobody has ever applied Physics-Informed Neural Networks to it.
Until now.

</div>

---

## 🔭 What This Is

This project applies **Physics-Informed Neural Networks (PINNs)** — specifically **SIREN** (sinusoidal representation networks) — to the **Wheeler-DeWitt (WDW) equation** in minisuperspace quantum cosmology.

This is, to our knowledge, **the first application of PINNs to the Wheeler-DeWitt equation**.

The Wheeler-DeWitt equation is the foundational equation of canonical quantum gravity. It describes the quantum state of the universe itself — the "wave function of the universe" `Ψ(a)`, where `a` is the scale factor (the size of the universe). It sits at the heart of one of the deepest unsolved problems in theoretical physics: reconciling General Relativity with Quantum Mechanics.

### The Open Problem We're Attacking

The WDW equation has a 50-year-old unresolved ambiguity: **operator ordering**. When General Relativity is canonically quantized, the kinetic term in the Hamiltonian is ambiguous — it can be ordered in multiple ways, each giving a physically different quantum theory of gravity:

```
d²Ψ/da²  +  (p/a) dΨ/da  −  U(a) Ψ  =  0
```

The parameter **`p`** encodes the ordering choice:
| Value | Ordering | Source |
|-------|----------|--------|
| `p = 0` | Simple ordering | Naïve quantization |
| `p = 1` | Laplace-Beltrami | Geometric / path integral |
| `p = 3` | Misner ordering | Misner (1972) |

No experiment can currently distinguish them. No consensus exists. This is a **genuine open problem in quantum gravity**.

We use a **Meta-PINN** — a neural network that takes `(a, p)` as input — to produce the **first continuous map of how the wave function of the universe changes with operator ordering**. This has never been done.

---

## 🏗️ Architecture

```
                    Wheeler-DeWitt Equation
                    -Ψ'' - (p/a)Ψ' + U(a)Ψ = 0
                           ↓
              ┌─────────────────────────────┐
              │         Meta-PINN           │
              │                             │
    a ──────→ │  SIREN  →  SIREN  →  SIREN │ → [Re(Ψ), Im(Ψ)]
    p ──────→ │  (ω₀=30, 256 dim, 6 layers) │
              └─────────────────────────────┘
                           ↓
                   Loss = L_pde + L_bc + L_norm
                           ↓
                   Adam → L-BFGS (curriculum)
```

### Why SIREN?

WDW solutions oscillate with frequency growing as `~a²` in the classically allowed region (`a > 1`). Standard neural networks with `tanh` or `ReLU` activations suffer from **spectral bias** — they learn low-frequency components first and fail to represent high-frequency oscillations. SIREN's sinusoidal activations `sin(ω₀ · Wx + b)` eliminate spectral bias by construction and have derivatives that are also SIREN networks — making the PDE residual computation exact within the same function class.

### Two Models

**`WDWNet`** — solves the 1D WDW equation for a fixed ordering `p`:
```python
model = WDWNet(hidden_dim=256, n_layers=5, omega_0=30.0)
# Input:  a ∈ [0.01, 4.0],  shape [N, 1]
# Output: [Re(Ψ), Im(Ψ)],   shape [N, 2]
```

**`MetaWDWNet`** — the novel contribution: solves for all orderings simultaneously:
```python
model = MetaWDWNet(hidden_dim=256, n_layers=6, omega_0=30.0)
# Input:  (a, p),           shape [N, 2]
# Output: [Re(Ψ), Im(Ψ)],   shape [N, 2]
# p can be any value in [-2, 4] — continuous interpolation
```

---

## 🧪 The Physics

### The Equation

In closed FRW minisuperspace with cosmological constant `Λ = 3` (Planck units), the WDW equation is:

```
−d²Ψ/da²  −  (p/a) dΨ/da  +  U(a)·Ψ  =  0

where:  U(a) = a² − a⁴
```

The potential `U(a)` changes sign at the **classical turning point `a = 1`**:
- `a < 1`: `U > 0` → classically forbidden region → solutions exponential (quantum tunneling)
- `a = 1`: `U = 0` → turning point → WKB breaks down
- `a > 1`: `U < 0` → classically allowed region → solutions oscillatory (classical universe)

### Boundary Conditions

We implement and compare **two competing proposals** — both unresolved in the literature:

| Proposal | Author | Condition at `a → 0` | Physical Interpretation |
|---|---|---|---|
| **Hartle-Hawking** | Hartle & Hawking (1983) | `Ψ` regular, `Ψ'(0) = 0` | No-boundary: universe has no initial edge |
| **Vilenkin** | Vilenkin (1986) | Outgoing wave only | Tunneling: universe nucleated from nothing |

### The WKB Amplitude (Validation Target)

In the classically allowed region, the analytic WKB prediction for the amplitude envelope is:

```
A(a; p)  ∝  a^(−(p+1)/2) · |U(a)|^(−1/4)
```

We use this as ground truth to validate the PINN's extracted amplitude and measure where quantum corrections to WKB become significant.

---

## 🚀 Quick Start

### Install

```bash
git clone https://github.com/satyam-das/wdw-pinn.git
cd wdw-pinn
pip install -r requirements.txt
```

### Run Validation (single ordering)

```python
from src.model import WDWNet
from src.train import train_wdw
from src.physics import solve_wdw_reference
import torch

# Train PINN for p=0 (simple ordering)
model = WDWNet(hidden_dim=256, n_layers=5, omega_0=30.0)
losses = train_wdw(model, p=0.0, n_epochs_adam=5000, n_epochs_lbfgs=500)

# Compare against scipy DOP853 reference
a_ref, psi_ref = solve_wdw_reference(p=0.0)
```

### Run Full Validation Suite

```bash
python run_validation.py
# Trains p=0, 1, 2 with full epochs
# Saves comparison plots to results/
```

### Run Tests

```bash
pytest tests/ -v
# 15 tests — all should pass
```

---

## 📁 Project Structure

```
wdw-pinn/
│
├── src/
│   ├── model.py       # SIREN architecture — WDWNet + MetaWDWNet
│   ├── physics.py     # WDW potential, WKB amplitude, scipy reference solver
│   ├── loss.py        # PDE residual, HH/Vilenkin BCs, normalization loss
│   ├── train.py       # Adam curriculum + L-BFGS fine-tuning + RAR sampling
│   └── experiment.py  # High-level experiment runner
│
├── tests/
│   ├── test_model.py   # SIREN architecture tests
│   ├── test_physics.py # WDW potential & WKB tests
│   └── test_loss.py    # Loss function correctness tests
│
├── results/           # Training plots, validation figures (gitignored by default)
├── docs/
│   └── superpowers/
│       └── plans/     # Implementation plan
│
├── run_validation.py  # Standalone validation script
└── requirements.txt
```

---

## 📊 Project Status

| Component | Status | Notes |
|---|---|---|
| SIREN architecture (`WDWNet`, `MetaWDWNet`) | ✅ Complete | 15/15 tests passing |
| Physics engine (WDW potential, WKB, scipy) | ✅ Complete | Validated analytically |
| Loss functions (PDE, HH, Vilenkin, norm) | ✅ Complete | Unit tested |
| Training loop (Adam + L-BFGS + RAR) | ✅ Complete | Curriculum learning |
| Validation vs. scipy DOP853 | 🔄 In Progress | Tuning L2 error threshold |
| Meta-PINN operator ordering sweep | ⏳ Upcoming | Novel experiment |
| Phase diagram `\|Ψ(a, p)\|` heatmap | ⏳ Upcoming | Paper Figure 1 |
| HH vs. Vilenkin comparison across `p` | ⏳ Upcoming | Paper Table 1 |

---

## 📖 Background & Motivation

### Why the Wheeler-DeWitt Equation?

The WDW equation is where General Relativity and Quantum Mechanics collide head-on. It is:
1. The quantum gravity equation with the most direct experimental connection (via inflationary cosmology)
2. One of the oldest unsolved problems in theoretical physics (DeWitt 1967, Wheeler 1968)
3. Computationally bottlenecked — existing methods (WKB, shooting methods) fail near turning points and in multi-field cases

### Why PINNs?

Physics-Informed Neural Networks (Raissi, Perdikaris & Karniadakis 2019) are neural networks trained to satisfy PDEs via automatic differentiation — the loss function includes the PDE residual directly. They:
- Handle singular points (like `a → 0`) more gracefully than finite-difference schemes
- Generalize across parameter spaces without re-solving
- Enable the **Meta-PINN** formulation: one network for all orderings simultaneously

### The Gap We're Filling

The PINN literature and the quantum cosmology literature do not overlap. Quantum cosmologists don't read PINN papers. PINN researchers don't know what the Wheeler-DeWitt equation is. **This project sits at the exact intersection of both worlds** — and that intersection has been completely empty until now.

---

## 🗺️ Roadmap

- [x] SIREN architecture implementation
- [x] Physics module (potential, WKB, reference solver)
- [x] Loss functions (PDE residual + boundary conditions)
- [x] Training loop (Adam → L-BFGS curriculum)
- [x] Unit test suite (15 tests)
- [ ] Validated PINN for p = 0, 1, 2 (L2 < 0.15 vs scipy)
- [ ] Meta-PINN training over `p ∈ [−2, 4]`
- [ ] Phase diagram: continuous `|Ψ(a, p)|` heatmap
- [ ] Hartle-Hawking vs. Vilenkin probability ratio across all `p`
- [ ] WKB amplitude comparison + deviation map
- [ ] 2D WDW with scalar field `φ`
- [ ] arXiv preprint submission

---

## 📚 Key References

| Paper | Why It Matters |
|---|---|
| DeWitt (1967), *Phys. Rev.* 160, 1113 | Original WDW equation |
| Hartle & Hawking (1983), *Phys. Rev. D* 28, 2960 | No-boundary wave function |
| Vilenkin (1986), *Phys. Rev. D* 33, 3560 | Tunneling wave function |
| Kiefer (2007), *Quantum Gravity*, Oxford | Standard WDW textbook |
| Raissi, Perdikaris & Karniadakis (2019), *J. Comp. Phys.* | Foundational PINN paper |
| Sitzmann et al. (2020), NeurIPS | SIREN sinusoidal networks |
| Udrescu & Tegmark (2020), *Science Advances* | AI Feynman — symbolic regression in physics |

---

## 👤 Author

**Satyam Das**
VIT Vellore, Computer Science & Engineering (2025)
Generative AI Developer · Researcher

*3 published papers in AI/ML · Expertise in PINNs, GNNs, multi-agent LLMs, RAG*

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

*"The universe is a quantum mechanical system. Its wave function exists. We are trying to find it."*
— inspired by J.A. Wheeler

**If you find this work interesting, star the repo ⭐ and reach out.**

</div>
