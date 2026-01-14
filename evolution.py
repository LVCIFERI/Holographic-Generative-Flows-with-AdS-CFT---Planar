"""
evolution.py

ODE Integration for UV-Stabilized Flow Matching
================================================

Document-faithful implementation of:

Section 4 (Full velocity with residual):
----------------------------------------
    V_θ(S, r) = V_KG(S, r) + R_θ(S, r)

where R_θ = (0, N_θ) only affects dΠ̃/dr.

Algorithm 2 (Sampling):
-----------------------
For sampling, we integrate from r_IR to r_UV:
    1. Sample S(r_IR) ~ μ_IR
    2. Integrate dS/dr = V_θ(S, r) from r_IR to r_UV
    3. Return y_gen = Readout(S(r_UV))

Available Solvers
-----------------
Standard (non-symplectic):
- EULER: First-order explicit Euler
- HEUN: Second-order Heun's method (explicit trapezoidal)
- MIDPOINT: Second-order midpoint method
- RK4: Fourth-order Runge-Kutta (default, recommended)

Symplectic (energy-conserving):
- LEAPFROG: Störmer-Verlet kick-drift-kick (separable Hamiltonians)
- IMPLICIT_MIDPOINT: Implicit midpoint (general Hamiltonians)

Symplectic integrators preserve phase-space volume and energy (up to discretization),
making them ideal for long integrations in Hamiltonian systems. They require evolving
canonical variables (Φ, P) where P = ω(r)Π is the canonical momentum.

The RK4 solver provides excellent accuracy for the smooth velocity fields
encountered in flow matching, while remaining computationally efficient.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

import torch

from ads_cft.networks import (
    BulkState,
    CanonicalState,
    state_add as _state_add,
    state_scale as _state_scale,
    state_sub as _state_sub,
    state_lerp as _state_lerp,
    canonical_add as _canonical_add,
    canonical_scale as _canonical_scale,
    canonical_sub as _canonical_sub,
    bulk_to_canonical,
    canonical_to_bulk,
)
from ads_cft.config import ODESolver, IntegratorConfig

Tensor = torch.Tensor


# =============================================================================
# Note: State arithmetic helpers are imported from networks.py
# =============================================================================


# =============================================================================
# ODE Integration (Document Algorithm 2)
# =============================================================================


def integrate_ode(
    drift_fn: Callable[[BulkState, Tensor], BulkState],
    y0: BulkState,
    r0: float,
    r1: float,
    config: IntegratorConfig,
) -> BulkState:
    """
    Fixed-step ODE integration for BulkState from r0 to r1.

    Document Algorithm 2: Integrate dS/dr = V_θ(S, r)

    This function implements fixed-step ODE integration using the
    specified solver method. The velocity field V_θ(S, r) is evaluated
    at each step to advance the state.

    Args:
        drift_fn: Velocity field V_θ(S, r) -> dS/dr
        y0: Initial state at r0
        r0: Start radius (typically r_IR)
        r1: End radius (typically r_UV)
        config: Integration configuration

    Returns:
        Final state at r1

    Example
    -------
    >>> def velocity(state, r):
    ...     return BulkState(phi_tilde=state.pi_tilde, pi_tilde=-state.phi_tilde)
    >>> y0 = BulkState(phi_tilde=torch.ones(32, 2), pi_tilde=torch.zeros(32, 2))
    >>> config = IntegratorConfig(solver=ODESolver.RK4, n_steps=100)
    >>> y1 = integrate_ode(velocity, y0, 0.0, 1.0, config)

    Notes
    -----
    The solver methods have the following properties:

    - EULER: O(h) local error, O(h^0) global error
    - HEUN: O(h^2) local error, O(h) global error
    - MIDPOINT: O(h^2) local error, O(h) global error
    - RK4: O(h^4) local error, O(h^3) global error

    where h is the step size (r1 - r0) / n_steps.
    """
    n_steps = config.n_steps
    h = (r1 - r0) / n_steps
    y = y0
    r = r0
    device = y0.phi_tilde.device
    dtype = y0.phi_tilde.dtype

    for _ in range(n_steps):
        r_t = torch.tensor(r, device=device, dtype=dtype)

        if config.solver == ODESolver.EULER:
            # First-order Euler: y_{n+1} = y_n + h * f(y_n, r_n)
            k1 = drift_fn(y, r_t)
            y = _state_add(y, _state_scale(k1, h))

        elif config.solver == ODESolver.HEUN:
            # Second-order Heun's method (explicit trapezoidal)
            # k1 = f(y_n, r_n)
            # y_pred = y_n + h * k1
            # k2 = f(y_pred, r_n + h)
            # y_{n+1} = y_n + (h/2) * (k1 + k2)
            k1 = drift_fn(y, r_t)
            y_pred = _state_add(y, _state_scale(k1, h))
            k2 = drift_fn(y_pred, torch.tensor(r + h, device=device, dtype=dtype))
            y = _state_add(y, _state_scale(_state_add(k1, k2), 0.5 * h))

        elif config.solver == ODESolver.MIDPOINT:
            # Second-order midpoint method
            # k1 = f(y_n, r_n)
            # y_mid = y_n + (h/2) * k1
            # k2 = f(y_mid, r_n + h/2)
            # y_{n+1} = y_n + h * k2
            k1 = drift_fn(y, r_t)
            y_mid = _state_add(y, _state_scale(k1, 0.5 * h))
            k2 = drift_fn(
                y_mid, torch.tensor(r + 0.5 * h, device=device, dtype=dtype)
            )
            y = _state_add(y, _state_scale(k2, h))

        elif config.solver == ODESolver.RK4:
            # Fourth-order Runge-Kutta
            # k1 = f(y_n, r_n)
            # k2 = f(y_n + (h/2)*k1, r_n + h/2)
            # k3 = f(y_n + (h/2)*k2, r_n + h/2)
            # k4 = f(y_n + h*k3, r_n + h)
            # y_{n+1} = y_n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
            k1 = drift_fn(y, r_t)
            k2 = drift_fn(
                _state_add(y, _state_scale(k1, 0.5 * h)),
                torch.tensor(r + 0.5 * h, device=device, dtype=dtype),
            )
            k3 = drift_fn(
                _state_add(y, _state_scale(k2, 0.5 * h)),
                torch.tensor(r + 0.5 * h, device=device, dtype=dtype),
            )
            k4 = drift_fn(
                _state_add(y, _state_scale(k3, h)),
                torch.tensor(r + h, device=device, dtype=dtype),
            )
            sum_k = _state_add(
                _state_add(k1, _state_scale(k2, 2.0)),
                _state_add(_state_scale(k3, 2.0), k4),
            )
            y = _state_add(y, _state_scale(sum_k, h / 6.0))

        else:
            raise ValueError(f"Unknown solver: {config.solver}")

        r += h

    return y


# =============================================================================
# Trajectory Recording (for visualization/debugging)
# =============================================================================


def integrate_with_trajectory(
    drift_fn: Callable[[BulkState, Tensor], BulkState],
    y0: BulkState,
    r0: float,
    r1: float,
    config: IntegratorConfig,
    record_every: int = 1,
) -> Tuple[BulkState, Tensor, Tensor, Tensor]:
    """
    Integrate ODE and record trajectory.

    This is useful for visualization and debugging of the bulk evolution.
    The trajectory shows how the state evolves from IR to UV.

    Args:
        drift_fn: Velocity field V_θ(S, r) -> dS/dr
        y0: Initial state at r0
        r0: Start radius
        r1: End radius
        config: Integration configuration
        record_every: Record state every N steps (default: 1)

    Returns:
        Tuple of (final_state, r_traj, phi_traj, pi_traj) where:
        - final_state: BulkState at r1
        - r_traj: Tensor of radius values, shape (n_recorded,)
        - phi_traj: Tensor of Φ̃ values, shape (n_recorded, B, C, *spatial)
        - pi_traj: Tensor of Π̃ values, shape (n_recorded, B, C, *spatial)

    Example
    -------
    >>> y1, r_traj, phi_traj, pi_traj = integrate_with_trajectory(
    ...     velocity, y0, 0.0, 1.0, config, record_every=10
    ... )
    >>> print(f"Recorded {len(r_traj)} states")
    """
    n_steps = config.n_steps
    h = (r1 - r0) / n_steps
    y = y0
    r = r0
    device = y0.phi_tilde.device
    dtype = y0.phi_tilde.dtype

    # Storage for trajectory
    r_list: List[float] = [r]
    phi_list: List[Tensor] = [y.phi_tilde.detach().clone()]
    pi_list: List[Tensor] = [y.pi_tilde.detach().clone()]

    for i in range(n_steps):
        r_t = torch.tensor(r, device=device, dtype=dtype)

        # Use the configured solver
        if config.solver == ODESolver.EULER:
            k1 = drift_fn(y, r_t)
            y = _state_add(y, _state_scale(k1, h))

        elif config.solver == ODESolver.HEUN:
            k1 = drift_fn(y, r_t)
            y_pred = _state_add(y, _state_scale(k1, h))
            k2 = drift_fn(y_pred, torch.tensor(r + h, device=device, dtype=dtype))
            y = _state_add(y, _state_scale(_state_add(k1, k2), 0.5 * h))

        elif config.solver == ODESolver.MIDPOINT:
            k1 = drift_fn(y, r_t)
            y_mid = _state_add(y, _state_scale(k1, 0.5 * h))
            k2 = drift_fn(
                y_mid, torch.tensor(r + 0.5 * h, device=device, dtype=dtype)
            )
            y = _state_add(y, _state_scale(k2, h))

        elif config.solver == ODESolver.RK4:
            k1 = drift_fn(y, r_t)
            k2 = drift_fn(
                _state_add(y, _state_scale(k1, 0.5 * h)),
                torch.tensor(r + 0.5 * h, device=device, dtype=dtype),
            )
            k3 = drift_fn(
                _state_add(y, _state_scale(k2, 0.5 * h)),
                torch.tensor(r + 0.5 * h, device=device, dtype=dtype),
            )
            k4 = drift_fn(
                _state_add(y, _state_scale(k3, h)),
                torch.tensor(r + h, device=device, dtype=dtype),
            )
            sum_k = _state_add(
                _state_add(k1, _state_scale(k2, 2.0)),
                _state_add(_state_scale(k3, 2.0), k4),
            )
            y = _state_add(y, _state_scale(sum_k, h / 6.0))

        else:
            raise ValueError(f"Unknown solver: {config.solver}")

        r += h

        # Record trajectory at specified intervals
        if (i + 1) % record_every == 0:
            r_list.append(r)
            phi_list.append(y.phi_tilde.detach().clone())
            pi_list.append(y.pi_tilde.detach().clone())

    # Convert lists to tensors
    r_traj = torch.tensor(r_list, device=device, dtype=dtype)
    phi_traj = torch.stack(phi_list, dim=0)
    pi_traj = torch.stack(pi_list, dim=0)

    return y, r_traj, phi_traj, pi_traj


# =============================================================================
# Adaptive Step Size Integration (Optional Advanced Feature)
# =============================================================================


def integrate_ode_adaptive(
    drift_fn: Callable[[BulkState, Tensor], BulkState],
    y0: BulkState,
    r0: float,
    r1: float,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    max_steps: int = 10000,
    initial_step: Optional[float] = None,
) -> BulkState:
    """
    Adaptive step-size ODE integration using embedded RK4(5).

    This uses the Dormand-Prince method (DOPRI5) with adaptive step size
    control for efficient and accurate integration.

    Args:
        drift_fn: Velocity field V_θ(S, r) -> dS/dr
        y0: Initial state at r0
        r0: Start radius
        r1: End radius
        rtol: Relative tolerance (default: 1e-5)
        atol: Absolute tolerance (default: 1e-8)
        max_steps: Maximum number of steps (default: 10000)
        initial_step: Initial step size (default: (r1-r0)/100)

    Returns:
        Final state at r1

    Raises:
        RuntimeError: If max_steps is exceeded

    Notes
    -----
    The adaptive method is useful when the velocity field varies significantly
    across the radial domain. For smooth velocity fields (typical in trained
    models), the fixed-step RK4 is often sufficient and faster.
    """
    device = y0.phi_tilde.device
    dtype = y0.phi_tilde.dtype

    # Dormand-Prince coefficients
    c = torch.tensor([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1], device=device, dtype=dtype)
    a = [
        [],
        [1 / 5],
        [3 / 40, 9 / 40],
        [44 / 45, -56 / 15, 32 / 9],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
    ]
    b5 = torch.tensor(
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        device=device,
        dtype=dtype,
    )
    b4 = torch.tensor(
        [
            5179 / 57600,
            0,
            7571 / 16695,
            393 / 640,
            -92097 / 339200,
            187 / 2100,
            1 / 40,
        ],
        device=device,
        dtype=dtype,
    )

    y = y0
    r = r0
    h = initial_step if initial_step is not None else (r1 - r0) / 100.0

    step_count = 0
    while r < r1:
        if step_count >= max_steps:
            raise RuntimeError(f"Adaptive integration exceeded {max_steps} steps")

        # Ensure we don't overshoot
        if r + h > r1:
            h = r1 - r

        # Compute RK stages
        k = []
        for i in range(7):
            r_i = r + c[i] * h
            y_i = y
            for j in range(i):
                if len(a[i]) > j and a[i][j] != 0:
                    y_i = _state_add(y_i, _state_scale(k[j], h * a[i][j]))
            k.append(drift_fn(y_i, torch.tensor(r_i, device=device, dtype=dtype)))

        # Compute 5th and 4th order solutions
        y5 = y
        y4 = y
        for i in range(7):
            if b5[i] != 0:
                y5 = _state_add(y5, _state_scale(k[i], h * b5[i]))
            if b4[i] != 0:
                y4 = _state_add(y4, _state_scale(k[i], h * b4[i]))

        # Error estimate
        err_phi = torch.max(torch.abs(y5.phi_tilde - y4.phi_tilde))
        err_pi = torch.max(torch.abs(y5.pi_tilde - y4.pi_tilde))
        err = max(err_phi.item(), err_pi.item())

        # Tolerance check
        scale_phi = atol + rtol * torch.max(torch.abs(y.phi_tilde))
        scale_pi = atol + rtol * torch.max(torch.abs(y.pi_tilde))
        scale = min(scale_phi.item(), scale_pi.item())
        err_ratio = err / scale

        if err_ratio <= 1.0:
            # Accept step
            y = y5
            r = r + h
            step_count += 1

        # Adjust step size
        if err_ratio > 0:
            h = h * min(2.0, max(0.1, 0.9 * err_ratio ** (-0.2)))
        else:
            h = h * 2.0

    return y

# =============================================================================
# Symplectic Integration (Energy-Conserving)
# =============================================================================


def integrate_symplectic(
    force_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    omega_fn: Callable[[Tensor], Tensor],
    y0: CanonicalState,
    r0: float,
    r1: float,
    config: IntegratorConfig,
) -> CanonicalState:
    """
    Symplectic integration for canonical Hamiltonian system (Φ, P).
    
    The equations of motion are:
        Φ' = ∂H/∂P = P / ω(r)
        P' = -∂H/∂Φ = F(Φ, r)
    
    where F is the "force" combining Klein-Gordon and learned potential:
        F = ω(r)(Δ_ĝΦ/f² - m²Φ) - ω(r) ∂U_θ/∂Φ
    
    This preserves phase-space volume and energy (up to discretization error).
    
    Args:
        force_fn: Force function F(Φ, P, r) -> P' contribution
        omega_fn: Slice weight ω(r) function
        y0: Initial canonical state at r0
        r0: Start radius
        r1: End radius
        config: Integration configuration
        
    Returns:
        Final canonical state at r1
        
    Note
    ----
    For LEAPFROG: Works best when residual is a potential (separable Hamiltonian).
    For IMPLICIT_MIDPOINT: Works for general Hamiltonians but requires iteration.
    """
    n_steps = config.n_steps
    h = (r1 - r0) / n_steps
    
    phi = y0.phi
    P = y0.P
    r = r0
    
    device = phi.device
    dtype = phi.dtype
    
    for _ in range(n_steps):
        r_t = torch.tensor(r, device=device, dtype=dtype)
        r_mid = torch.tensor(r + 0.5 * h, device=device, dtype=dtype)
        r_next = torch.tensor(r + h, device=device, dtype=dtype)
        
        if config.solver == ODESolver.LEAPFROG:
            # Störmer-Verlet (Kick-Drift-Kick)
            # 1. Half kick: P_{n+1/2} = P_n + (h/2) F(Φ_n, r_n)
            # 2. Full drift: Φ_{n+1} = Φ_n + h P_{n+1/2} / ω(r_{n+1/2})
            # 3. Half kick: P_{n+1} = P_{n+1/2} + (h/2) F(Φ_{n+1}, r_{n+1})
            
            # Half kick
            F_n = force_fn(phi, P, r_t)
            P_half = P + 0.5 * h * F_n
            
            # Full drift (use ω at midpoint for accuracy)
            omega_mid = omega_fn(r_mid)
            if omega_mid.ndim < phi.ndim:
                omega_mid = omega_mid.view(*([1] * (phi.ndim - omega_mid.ndim)), *omega_mid.shape)
                omega_mid = omega_mid.expand_as(phi)
            omega_safe = torch.clamp(omega_mid, min=1e-10)
            phi = phi + h * P_half / omega_safe
            
            # Half kick
            F_next = force_fn(phi, P_half, r_next)
            P = P_half + 0.5 * h * F_next
            
        elif config.solver == ODESolver.IMPLICIT_MIDPOINT:
            # Implicit midpoint: S_{n+1} = S_n + h V((S_n + S_{n+1})/2, r_{n+1/2})
            # Solve via fixed-point iteration
            
            phi_next = phi.clone()
            P_next = P.clone()
            
            omega_mid = omega_fn(r_mid)
            if omega_mid.ndim < phi.ndim:
                omega_mid = omega_mid.view(*([1] * (phi.ndim - omega_mid.ndim)), *omega_mid.shape)
                omega_mid = omega_mid.expand_as(phi)
            omega_safe = torch.clamp(omega_mid, min=1e-10)
            
            for _ in range(config.implicit_max_iter):
                # Midpoint values
                phi_mid = 0.5 * (phi + phi_next)
                P_mid = 0.5 * (P + P_next)
                
                # Velocity at midpoint
                F_mid = force_fn(phi_mid, P_mid, r_mid)
                
                # Update
                phi_new = phi + h * P_mid / omega_safe
                P_new = P + h * F_mid
                
                # Check convergence
                err_phi = torch.max(torch.abs(phi_new - phi_next))
                err_P = torch.max(torch.abs(P_new - P_next))
                
                phi_next = phi_new
                P_next = P_new
                
                if max(err_phi.item(), err_P.item()) < config.implicit_tol:
                    break
            
            phi = phi_next
            P = P_next
        
        else:
            raise ValueError(
                f"Solver {config.solver} is not symplectic. "
                f"Use LEAPFROG or IMPLICIT_MIDPOINT for symplectic integration."
            )
        
        r += h
    
    return CanonicalState(phi=phi, P=P)


def integrate_ode_general(
    drift_fn: Callable[[BulkState, Tensor], BulkState],
    y0: BulkState,
    r0: float,
    r1: float,
    config: IntegratorConfig,
    # Symplectic-specific arguments (optional)
    force_fn: Optional[Callable[[Tensor, Tensor, Tensor], Tensor]] = None,
    omega_fn: Optional[Callable[[Tensor], Tensor]] = None,
    delta: Optional[Tensor] = None,
    geometry=None,
) -> BulkState:
    """
    General ODE integration supporting both standard and symplectic solvers.
    
    This is the unified entry point that routes to the appropriate integrator
    based on the solver type in config.
    
    For standard solvers (EULER, HEUN, MIDPOINT, RK4):
        Uses integrate_ode with drift_fn
        
    For symplectic solvers (LEAPFROG, IMPLICIT_MIDPOINT):
        Converts to canonical variables, uses integrate_symplectic,
        then converts back to bulk variables.
        Requires force_fn, omega_fn, and delta to be provided.
    
    Args:
        drift_fn: Velocity field V_θ(S, r) -> dS/dr (for standard solvers)
        y0: Initial UV-stabilized state at r0
        r0: Start radius
        r1: End radius
        config: Integration configuration
        force_fn: Force F(Φ, P, r) (required for symplectic)
        omega_fn: Slice weight ω(r) (required for symplectic)
        delta: Conformal dimensions (required for symplectic)
        geometry: AdS geometry object (required for HSV symplectic)
        
    Returns:
        Final UV-stabilized state at r1
    """
    # Standard solvers
    if config.solver in {ODESolver.EULER, ODESolver.HEUN, ODESolver.MIDPOINT, ODESolver.RK4}:
        return integrate_ode(drift_fn, y0, r0, r1, config)
    
    # Symplectic solvers
    if config.solver in {ODESolver.LEAPFROG, ODESolver.IMPLICIT_MIDPOINT}:
        if force_fn is None or omega_fn is None or delta is None:
            raise ValueError(
                f"Symplectic solver {config.solver} requires force_fn, omega_fn, and delta. "
                f"Use ResidualType.POTENTIAL for symplectic integration."
            )
        
        device = y0.phi_tilde.device
        dtype = y0.phi_tilde.dtype
        
        # Convert to canonical at r0
        r0_t = torch.tensor(r0, device=device, dtype=dtype)
        omega_0 = omega_fn(r0_t)
        y0_canonical = bulk_to_canonical(y0, r0_t, delta, omega_0, geometry=geometry)
        
        # Integrate in canonical variables
        y1_canonical = integrate_symplectic(force_fn, omega_fn, y0_canonical, r0, r1, config)
        
        # Convert back to bulk at r1
        r1_t = torch.tensor(r1, device=device, dtype=dtype)
        omega_1 = omega_fn(r1_t)
        y1 = canonical_to_bulk(y1_canonical, r1_t, delta, omega_1, geometry=geometry)
        
        return y1
    
    raise ValueError(f"Unknown solver: {config.solver}")


# =============================================================================
# Symplectic Integration with Trajectory Recording
# =============================================================================


def integrate_symplectic_with_trajectory(
    force_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
    omega_fn: Callable[[Tensor], Tensor],
    y0: CanonicalState,
    r0: float,
    r1: float,
    config: IntegratorConfig,
    record_every: int = 1,
) -> Tuple[CanonicalState, Tensor, Tensor, Tensor]:
    """
    Symplectic integration with trajectory recording.
    
    Same as integrate_symplectic but records the state at intervals.
    Useful for visualization and energy conservation verification.
    
    Args:
        force_fn: Force function F(Φ, P, r)
        omega_fn: Slice weight ω(r) function
        y0: Initial canonical state
        r0: Start radius
        r1: End radius
        config: Integration configuration
        record_every: Record every N steps
        
    Returns:
        Tuple of (final_state, r_traj, phi_traj, P_traj)
    """
    n_steps = config.n_steps
    h = (r1 - r0) / n_steps
    
    phi = y0.phi
    P = y0.P
    r = r0
    
    device = phi.device
    dtype = phi.dtype
    
    # Storage
    r_list: List[float] = [r]
    phi_list: List[Tensor] = [phi.detach().clone()]
    P_list: List[Tensor] = [P.detach().clone()]
    
    for i in range(n_steps):
        r_t = torch.tensor(r, device=device, dtype=dtype)
        r_mid = torch.tensor(r + 0.5 * h, device=device, dtype=dtype)
        r_next = torch.tensor(r + h, device=device, dtype=dtype)
        
        if config.solver == ODESolver.LEAPFROG:
            # Kick-Drift-Kick
            F_n = force_fn(phi, P, r_t)
            P_half = P + 0.5 * h * F_n
            
            omega_mid = omega_fn(r_mid)
            if omega_mid.ndim < phi.ndim:
                omega_mid = omega_mid.view(*([1] * (phi.ndim - omega_mid.ndim)), *omega_mid.shape)
                omega_mid = omega_mid.expand_as(phi)
            omega_safe = torch.clamp(omega_mid, min=1e-10)
            phi = phi + h * P_half / omega_safe
            
            F_next = force_fn(phi, P_half, r_next)
            P = P_half + 0.5 * h * F_next
            
        elif config.solver == ODESolver.IMPLICIT_MIDPOINT:
            phi_next = phi.clone()
            P_next = P.clone()
            
            omega_mid = omega_fn(r_mid)
            if omega_mid.ndim < phi.ndim:
                omega_mid = omega_mid.view(*([1] * (phi.ndim - omega_mid.ndim)), *omega_mid.shape)
                omega_mid = omega_mid.expand_as(phi)
            omega_safe = torch.clamp(omega_mid, min=1e-10)
            
            for _ in range(config.implicit_max_iter):
                phi_mid = 0.5 * (phi + phi_next)
                P_mid = 0.5 * (P + P_next)
                F_mid = force_fn(phi_mid, P_mid, r_mid)
                
                phi_new = phi + h * P_mid / omega_safe
                P_new = P + h * F_mid
                
                err = max(
                    torch.max(torch.abs(phi_new - phi_next)).item(),
                    torch.max(torch.abs(P_new - P_next)).item()
                )
                
                phi_next = phi_new
                P_next = P_new
                
                if err < config.implicit_tol:
                    break
            
            phi = phi_next
            P = P_next
        
        r += h
        
        if (i + 1) % record_every == 0:
            r_list.append(r)
            phi_list.append(phi.detach().clone())
            P_list.append(P.detach().clone())
    
    r_traj = torch.tensor(r_list, device=device, dtype=dtype)
    phi_traj = torch.stack(phi_list, dim=0)
    P_traj = torch.stack(P_list, dim=0)
    
    return CanonicalState(phi=phi, P=P), r_traj, phi_traj, P_traj


# =============================================================================
# Energy/Hamiltonian Monitoring (for symplectic verification)
# =============================================================================


def compute_hamiltonian(
    phi: Tensor,
    P: Tensor,
    omega: Tensor,
    laplacian_eigenvalues: Optional[Tensor] = None,
    m_squared: Optional[Tensor] = None,
    f: float = 1.0,
) -> Tensor:
    """
    Compute the Hamiltonian H = kinetic + potential.
    
    H = ∫ [P²/(2ω) + ω/2 (|∇Φ|²/f² + m²Φ²)] dvol
    
    For diagonal Laplacian (Fourier modes), this simplifies to:
        H = Σ_n [P_n²/(2ω) + ω/2 (λ_n/f² + m²) Φ_n²]
    
    Args:
        phi: Field values (B, C, ...) or (B, C)
        P: Canonical momenta (B, C, ...) or (B, C)
        omega: Slice weight ω(r)
        laplacian_eigenvalues: λ_n for diagonal case (optional)
        m_squared: Mass squared m² (optional)
        f: Warp factor (default: 1)
        
    Returns:
        Hamiltonian per sample, shape (B,)
    """
    # Expand omega
    if omega.ndim < phi.ndim:
        omega = omega.view(*([1] * (phi.ndim - omega.ndim)), *omega.shape)
        omega = omega.expand_as(phi)
    omega_safe = torch.clamp(omega, min=1e-10)
    
    # Kinetic: P²/(2ω)
    kinetic = (P ** 2) / (2 * omega_safe)
    
    # Potential: ω/2 (λ/f² + m²) Φ²
    potential_coeff = omega_safe / 2
    if laplacian_eigenvalues is not None:
        # Add gradient term
        lam = laplacian_eigenvalues
        if lam.ndim < phi.ndim:
            lam = lam.view(*([1] * (phi.ndim - lam.ndim)), *lam.shape)
            lam = lam.expand_as(phi)
        potential_coeff = potential_coeff * (lam / (f ** 2))
    if m_squared is not None:
        m2 = m_squared
        if m2.ndim < phi.ndim:
            m2 = m2.view(*([1] * (phi.ndim - m2.ndim)), *m2.shape)
            m2 = m2.expand_as(phi)
        potential_coeff = potential_coeff + omega_safe / 2 * m2
    
    potential = potential_coeff * (phi ** 2)
    
    # Sum over all dimensions except batch
    H = (kinetic + potential).sum(dim=tuple(range(1, phi.ndim)))
    
    return H
