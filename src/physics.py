# src/physics.py
import numpy as np
from scipy.integrate import solve_ivp


def wdw_potential(a: np.ndarray) -> np.ndarray:
    """
    Wheeler-DeWitt potential U(a) = a^2 - a^4
    for closed FRW minisuperspace with Λ=3.

    Positive (classically forbidden) for a < 1.
    Negative (classically allowed) for a > 1.
    Turning point at a = 1.
    """
    return a**2 - a**4


def wkb_amplitude(a: np.ndarray, p: float = 0.0) -> np.ndarray:
    """
    WKB amplitude in classically allowed region (a > 1):
        A(a; p) ∝ a^(-(p+1)/2) · |U(a)|^(-1/4)

    Only valid for a > 1 where U(a) < 0.
    """
    if np.any(a <= 1.0):
        raise ValueError("wkb_amplitude requires a > 1 (classically allowed region only)")

    U = wdw_potential(a)
    envelope = np.abs(U) ** (-0.25)
    ordering_factor = a ** (-(p + 1) / 2)
    amp = envelope * ordering_factor
    return amp / (amp[0] + 1e-12)


def solve_wdw_reference(
    p: float = 0.0,
    a_min: float = 0.01,
    a_max: float = 3.5,
    n_points: int = 500,
) -> tuple:
    """
    Solve the WDW equation as an ODE using scipy (DOP853).

    WDW ODE: Ψ'' + (p/a)Ψ' - U(a)Ψ = 0
    Rewritten as system: y[0] = Ψ, y[1] = Ψ'

    Returns: (a_grid, Ψ_real)
    """
    def rhs(a, y):
        psi, dpsi = y
        # Start from a_min > 0 to avoid the singular coefficient (p/a) at a=0.
        # Frobenius series gives Ψ(a_min)=1, Ψ'(a_min)=0 for the regular solution branch.
        a_safe = max(a, 1e-8)
        U = wdw_potential(np.array([a_safe]))[0]
        d2psi = -(p / a_safe) * dpsi + U * psi
        return [dpsi, d2psi]

    a_span = (a_min, a_max)
    y0 = [1.0, 0.0]

    sol = solve_ivp(
        rhs,
        a_span,
        y0,
        method="DOP853",
        dense_output=True,
        rtol=1e-10,
        atol=1e-12,
    )

    a_grid = np.linspace(a_min, a_max, n_points)
    psi = sol.sol(a_grid)[0]
    return a_grid, psi
