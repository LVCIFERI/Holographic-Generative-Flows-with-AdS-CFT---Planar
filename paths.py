"""
paths.py

Path Interpolation for Bulk Flow Matching
==========================================

This module provides path interpolation methods for connecting IR prior
states to UV data states in the bulk flow matching framework.

Document Section 9: Flow Matching Paths
---------------------------------------

Two primary path types are supported:

1. LINEAR path (robust default):
   - S_t = (1-t)S_0 + tS_1
   - U_t = S_1 - S_0 (constant target velocity)
   - Simple, stable, works well in practice
   - Does not account for AdS geometry

2. ADS_AFFINE_HERMITE path (geometric):
   - Uses affine parameter u(r) = (1/Z) ∫_{r_IR}^r 1/ω(s) ds
   - Cubic Hermite interpolation in u-space
   - Respects the AdS geometry through slice weights
   - Has Π̃_t = ∂_r Φ̃_t by construction (matches backbone)

The Hermite path is the document-faithful choice that fully leverages
the AdS structure. The linear path is provided as a simpler alternative
that still works well in many cases.

Document References
-------------------
- eq (rectified-path): Hermite path definition
- eq (target-velocity): Target velocity for Hermite path
- Section 9: Path comparison and recommendations
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn

from ads_cft.networks import BulkState
from ads_cft.config import PathType, PathConfig

Tensor = torch.Tensor


# =============================================================================
# Linear Path Interpolation
# =============================================================================


def linear_interpolation(
    S0: BulkState,
    S1: BulkState,
    t: Tensor,
) -> BulkState:
    """
    Linear path interpolation: S_t = (1-t)S_0 + tS_1.

    Args:
        S0: IR state (Φ̃_0, Π̃_0)
        S1: UV state (Φ̃_1, Π̃_1)
        t: Time in [0, 1], shape (B,) or scalar

    Returns:
        Interpolated state S_t

    Example
    -------
    >>> S0 = BulkState(phi_tilde=torch.randn(32, 2), pi_tilde=torch.randn(32, 2))
    >>> S1 = BulkState(phi_tilde=torch.randn(32, 2), pi_tilde=torch.randn(32, 2))
    >>> t = torch.tensor(0.5)
    >>> S_t = linear_interpolation(S0, S1, t)
    """
    t_b = _broadcast_time(t, S0.phi_tilde)

    phi_t = (1 - t_b) * S0.phi_tilde + t_b * S1.phi_tilde
    pi_t = (1 - t_b) * S0.pi_tilde + t_b * S1.pi_tilde

    return BulkState(phi_tilde=phi_t, pi_tilde=pi_t)


def linear_velocity(
    S0: BulkState,
    S1: BulkState,
) -> BulkState:
    """
    Linear path target velocity: U_t = S_1 - S_0 (constant).

    Args:
        S0: IR state (Φ̃_0, Π̃_0)
        S1: UV state (Φ̃_1, Π̃_1)

    Returns:
        Target velocity U_t (constant in t)
    """
    return BulkState(
        phi_tilde=S1.phi_tilde - S0.phi_tilde,
        pi_tilde=S1.pi_tilde - S0.pi_tilde,
    )


def linear_path_and_velocity(
    S0: BulkState,
    S1: BulkState,
    t: Tensor,
) -> Tuple[BulkState, BulkState]:
    """
    Compute both interpolated state and target velocity for linear path.

    Args:
        S0: IR state (Φ̃_0, Π̃_0)
        S1: UV state (Φ̃_1, Π̃_1)
        t: Time in [0, 1], shape (B,) or scalar

    Returns:
        Tuple of (S_t, U_t) where U_t = S_1 - S_0
    """
    S_t = linear_interpolation(S0, S1, t)
    U_t = linear_velocity(S0, S1)
    return S_t, U_t


# =============================================================================
# Hermite Basis Functions
# =============================================================================


def _hermite_basis(u: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Cubic Hermite basis polynomials on [0, 1].

    Standard Hermite interpolation basis:
        h00(u) = 2u³ - 3u² + 1   (value at u=0)
        h10(u) = u³ - 2u² + u    (derivative at u=0)
        h01(u) = -2u³ + 3u²      (value at u=1)
        h11(u) = u³ - u²         (derivative at u=1)

    Properties:
        h00(0) = 1, h00(1) = 0, h00'(0) = 0, h00'(1) = 0
        h10(0) = 0, h10(1) = 0, h10'(0) = 1, h10'(1) = 0
        h01(0) = 0, h01(1) = 1, h01'(0) = 0, h01'(1) = 0
        h11(0) = 0, h11(1) = 0, h11'(0) = 0, h11'(1) = 1

    Args:
        u: Parameter in [0, 1]

    Returns:
        Tuple of (h00, h10, h01, h11)
    """
    u2 = u ** 2
    u3 = u ** 3

    h00 = 2 * u3 - 3 * u2 + 1
    h10 = u3 - 2 * u2 + u
    h01 = -2 * u3 + 3 * u2
    h11 = u3 - u2

    return h00, h10, h01, h11


def _hermite_basis_deriv(u: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    First derivative of cubic Hermite basis.

    Args:
        u: Parameter in [0, 1]

    Returns:
        Tuple of (dh00/du, dh10/du, dh01/du, dh11/du)
    """
    u2 = u ** 2

    dh00 = 6 * u2 - 6 * u
    dh10 = 3 * u2 - 4 * u + 1
    dh01 = -6 * u2 + 6 * u
    dh11 = 3 * u2 - 2 * u

    return dh00, dh10, dh01, dh11


def _hermite_basis_second_deriv(u: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Second derivative of cubic Hermite basis.

    Args:
        u: Parameter in [0, 1]

    Returns:
        Tuple of (d²h00/du², d²h10/du², d²h01/du², d²h11/du²)
    """
    d2h00 = 12 * u - 6
    d2h10 = 6 * u - 4
    d2h01 = -12 * u + 6
    d2h11 = 6 * u - 2

    return d2h00, d2h10, d2h01, d2h11


# =============================================================================
# Time Broadcasting Helper
# =============================================================================


def _broadcast_time(t: Tensor, like: Tensor) -> Tensor:
    """
    Broadcast time t across spatial/channel dims of like tensor.

    Args:
        t: Time tensor (scalar or (B,))
        like: Target tensor to match shape

    Returns:
        Broadcasted time tensor
    """
    if t.ndim == 0:
        t = t.expand(like.shape[0])
    while t.ndim < like.ndim:
        t = t.unsqueeze(-1)
    return t


# =============================================================================
# AdS-Affine Hermite Path
# NOTE: This class is equivalent to AdSAffineHermiteCoupling in networks.py.
# This is the canonical implementation; the one in networks.py exists for
# backwards compatibility and to avoid circular imports.
# =============================================================================


class AdSAffineHermitePath(nn.Module):
    """
    AdS-affine cubic Hermite path in UV-stable variables.

    Document eq (rectified-path):
    Uses the affine parameter u(r) = (1/Z) ∫_{r_IR}^r 1/ω(s) ds
    for cubic Hermite interpolation between endpoints.

    NOTE: This class is equivalent to AdSAffineHermiteCoupling in networks.py.

    The affine parameter accounts for the varying slice measure ω(r) = f(r)^d,
    so that the path progresses uniformly in the intrinsic geometry.

    The induced Π̃(t) and target velocity are computed via chain rule
    as specified in eq (target-velocity).

    Key property: By construction, Π̃_t = ∂_r Φ̃_t, which exactly matches
    the Klein-Gordon backbone's dΦ̃/dr = Π̃. This means the phi residual
    is always zero for Hermite path.

    Attributes
    ----------
    geometry : AdSGeometry
        AdS geometry with warp factor f(r)
    r_ir : float
        IR boundary radius
    r_uv : float
        UV boundary radius

    Example
    -------
    >>> path = AdSAffineHermitePath(geometry, r_ir=0.0, r_uv=1.0)
    >>> S_t, U_t = path.state_and_target_velocity(S0, S1, t)
    """

    def __init__(
        self,
        geometry,  # AdSGeometry
        r_ir: float,
        r_uv: float,
        use_radial_hermite: bool = False,
    ) -> None:
        """
        Initialize AdS-affine Hermite path.

        Args:
            geometry: AdS geometry object
            r_ir: IR boundary radius
            r_uv: UV boundary radius
            use_radial_hermite: If True, use radial Hermite (u=t, no ω factors).
                Auto-enabled for HSV geometries. Default False for standard AdS.
        """
        super().__init__()
        self.geometry = geometry
        self.r_ir = float(r_ir)
        self.r_uv = float(r_uv)
        self.use_radial_hermite = use_radial_hermite

    @property
    def r_span(self) -> float:
        """Return radial span r_UV - r_IR."""
        return self.r_uv - self.r_ir

    def state_and_target_velocity(
        self,
        state0: BulkState,
        state1: BulkState,
        t: Tensor,
    ) -> Tuple[BulkState, BulkState]:
        """
        Compute interpolated state and target velocity at time t.

        Document eq (rectified-path) and (target-velocity).

        For HSV geometries, uses "radial Hermite" (u=t, dr/du=r_span) instead
        of affine-parameter Hermite to avoid numerical instabilities from
        exploding/vanishing ω factors.

        The Hermite interpolation uses:
        - Endpoint values: Φ̃_0, Φ̃_1
        - Endpoint tangents: m_0 = (dr/du)|_{u=0} Π̃_0, m_1 = (dr/du)|_{u=1} Π̃_1

        Args:
            state0: IR state (Φ̃_0, Π̃_0)
            state1: UV state (Φ̃_1, Π̃_1)
            t: Normalized time in [0, 1]

        Returns:
            Tuple of (state_t, target_velocity_t)
        """
        device = state0.phi_tilde.device
        dtype = state0.phi_tilde.dtype
        t = torch.as_tensor(t, device=device, dtype=dtype)

        # Physical radius r(t) = r_IR + t * (r_UV - r_IR)
        t_b = _broadcast_time(t, state0.phi_tilde)
        r_ir_t = torch.tensor(self.r_ir, device=device, dtype=dtype)
        r_uv_t = torch.tensor(self.r_uv, device=device, dtype=dtype)
        r_t = r_ir_t + self.r_span * t_b
        rspan = torch.tensor(self.r_span, device=device, dtype=dtype)

        # Check if HSV geometry or use_radial_hermite - use radial Hermite instead of affine Hermite
        is_hsv = getattr(self.geometry, 'is_hsv', False)
        use_radial = is_hsv or self.use_radial_hermite

        if use_radial:
            # =================================================================
            # HSV: RADIAL HERMITE PATH
            # =================================================================
            # For HSV, ω(r) has inverted behavior (decreases toward UV),
            # making affine-parameter formulation numerically unstable.
            # Instead, use t directly as the Hermite parameter:
            #   - u = t (normalized radius IS the parameter)
            #   - dr/du = r_span (constant)
            #   - m₀ = r_span * Π̃₀, m₁ = r_span * Π̃₁
            # This removes all ω factors and stays stable for any p.
            # =================================================================
            
            u_b = t_b  # u = t for radial Hermite
            
            # Constant tangent scaling (no ω factors!)
            m0 = state0.pi_tilde * rspan
            m1 = state1.pi_tilde * rspan

            # Hermite interpolation in t-space
            h00, h10, h01, h11 = _hermite_basis(u_b)
            phi_t = (
                h00 * state0.phi_tilde
                + h10 * m0
                + h01 * state1.phi_tilde
                + h11 * m1
            )

            # First derivative: dΦ̃/dt
            dh00, dh10, dh01, dh11 = _hermite_basis_deriv(u_b)
            phi_dot = (
                dh00 * state0.phi_tilde
                + dh10 * m0
                + dh01 * state1.phi_tilde
                + dh11 * m1
            )

            # Second derivative: d²Φ̃/dt²
            d2h00, d2h10, d2h01, d2h11 = _hermite_basis_second_deriv(u_b)
            phi_ddot = (
                d2h00 * state0.phi_tilde
                + d2h10 * m0
                + d2h01 * state1.phi_tilde
                + d2h11 * m1
            )

            # Π̃_t = dΦ̃/dr = (dΦ̃/dt) / (dr/dt) = φ̇ / r_span
            pi_t = phi_dot / rspan

            # Target velocities
            # dΦ̃/dt = φ̇ (direct from Hermite)
            dphi_dt = phi_dot

            # dΠ̃/dt = d(φ̇/r_span)/dt = φ̈ / r_span
            dpi_dt = phi_ddot / rspan

        else:
            # =================================================================
            # STANDARD AdS: AFFINE HERMITE PATH
            # =================================================================
            # Affine parameter u(r_t) ∈ [0, 1]
            u = self.geometry.affine_parameter(r_t, self.r_ir, self.r_uv)
            if hasattr(u, "to"):
                u = u.to(device=device, dtype=dtype)
            elif not isinstance(u, Tensor):
                u = torch.tensor(u, device=device, dtype=dtype)
            u_b = _broadcast_time(u, state0.phi_tilde)

            # Endpoint tangents: dΦ̃/du = (dr/du) Π̃ = Z ω(r) Π̃
            drdu_ir = self.geometry.dr_du(r_ir_t, self.r_ir, self.r_uv)
            drdu_uv = self.geometry.dr_du(r_uv_t, self.r_ir, self.r_uv)

            # Ensure dr_du results are on correct device
            if hasattr(drdu_ir, "to"):
                drdu_ir = drdu_ir.to(device=device, dtype=dtype)
            else:
                drdu_ir = torch.tensor(drdu_ir, device=device, dtype=dtype)
            if hasattr(drdu_uv, "to"):
                drdu_uv = drdu_uv.to(device=device, dtype=dtype)
            else:
                drdu_uv = torch.tensor(drdu_uv, device=device, dtype=dtype)

            m0 = state0.pi_tilde * drdu_ir  # dΦ̃/du at u=0
            m1 = state1.pi_tilde * drdu_uv  # dΦ̃/du at u=1

            # Hermite interpolation in u
            h00, h10, h01, h11 = _hermite_basis(u_b)
            phi_t = (
                h00 * state0.phi_tilde
                + h10 * m0
                + h01 * state1.phi_tilde
                + h11 * m1
            )

            # First u-derivative: dΦ̃/du
            dh00, dh10, dh01, dh11 = _hermite_basis_deriv(u_b)
            phi_u = (
                dh00 * state0.phi_tilde
                + dh10 * m0
                + dh01 * state1.phi_tilde
                + dh11 * m1
            )

            # Second u-derivative: d²Φ̃/du²
            d2h00, d2h10, d2h01, d2h11 = _hermite_basis_second_deriv(u_b)
            phi_uu = (
                d2h00 * state0.phi_tilde
                + d2h10 * m0
                + d2h01 * state1.phi_tilde
                + d2h11 * m1
            )

            # Geometry factors
            omega = self.geometry.slice_weight(r_t)
            omega_prime = self.geometry.slice_weight_prime(r_t)
            Z = self.geometry.affine_norm_const(
                self.r_ir, self.r_uv, device=device, dtype=dtype
            )

            # Ensure geometry factors are on correct device
            if hasattr(omega, "to"):
                omega = omega.to(device=device, dtype=dtype)
            elif not isinstance(omega, Tensor):
                omega = torch.tensor(omega, device=device, dtype=dtype)
            if hasattr(omega_prime, "to"):
                omega_prime = omega_prime.to(device=device, dtype=dtype)
            elif not isinstance(omega_prime, Tensor):
                omega_prime = torch.tensor(omega_prime, device=device, dtype=dtype)

            # Broadcast scalars
            omega = _broadcast_time(omega, state0.phi_tilde)
            omega_prime = _broadcast_time(omega_prime, state0.phi_tilde)

            # Π̃_t = ∂_r Φ̃ = (du/dr) dΦ̃/du = dΦ̃/du / (Z ω)
            pi_t = phi_u / (Z * omega)

            # Target velocity in t-time (Document eq target-velocity)
            # dΦ̃/dt = (r_UV - r_IR) Π̃_t
            dphi_dt = rspan * pi_t

            # dΠ̃/dt = (r_UV - r_IR) [d²Φ̃/du² / (Z²ω²) - ω' dΦ̃/du / (Z ω²)]
            dpi_dt = rspan * (
                phi_uu / (Z * Z * omega * omega)
                - omega_prime * phi_u / (Z * omega * omega)
            )

        state_t = BulkState(phi_tilde=phi_t, pi_tilde=pi_t)
        target_vel = BulkState(phi_tilde=dphi_dt, pi_tilde=dpi_dt)

        return state_t, target_vel

    def forward(
        self,
        state0: BulkState,
        state1: BulkState,
        t: Tensor,
    ) -> BulkState:
        """
        Return interpolated state at time t.

        Args:
            state0: IR state
            state1: UV state
            t: Normalized time

        Returns:
            Interpolated state
        """
        state_t, _ = self.state_and_target_velocity(state0, state1, t)
        return state_t


# =============================================================================
# Quadratic Path (Optional Alternative)
# =============================================================================


class QuadraticPath(nn.Module):
    """
    Quadratic path with learnable or fixed midpoint.

    S_t = (1-t)² S_0 + 2t(1-t) S_mid + t² S_1

    This provides a smooth path through a midpoint that can be
    learned or set to interpolate more naturally.

    Attributes
    ----------
    midpoint_weight : float
        How much to blend the prior midpoint (0.5 * (S0 + S1))
        with an offset. Default 1.0 means pure midpoint.
    """

    def __init__(self, midpoint_weight: float = 1.0) -> None:
        """
        Initialize quadratic path.

        Args:
            midpoint_weight: Weight for midpoint interpolation
        """
        super().__init__()
        self.midpoint_weight = midpoint_weight

    def state_and_target_velocity(
        self,
        state0: BulkState,
        state1: BulkState,
        t: Tensor,
    ) -> Tuple[BulkState, BulkState]:
        """
        Compute quadratic interpolation and velocity.

        Args:
            state0: Start state
            state1: End state
            t: Time in [0, 1]

        Returns:
            Tuple of (S_t, U_t)
        """
        t_b = _broadcast_time(t, state0.phi_tilde)

        # Midpoint (simple average)
        phi_mid = 0.5 * (state0.phi_tilde + state1.phi_tilde)
        pi_mid = 0.5 * (state0.pi_tilde + state1.pi_tilde)

        # Bernstein polynomials for quadratic
        b0 = (1 - t_b) ** 2
        b1 = 2 * t_b * (1 - t_b)
        b2 = t_b ** 2

        # Interpolation
        phi_t = b0 * state0.phi_tilde + b1 * phi_mid + b2 * state1.phi_tilde
        pi_t = b0 * state0.pi_tilde + b1 * pi_mid + b2 * state1.pi_tilde

        # Derivatives of Bernstein polynomials
        db0 = -2 * (1 - t_b)
        db1 = 2 * (1 - 2 * t_b)
        db2 = 2 * t_b

        # Velocity
        dphi_dt = db0 * state0.phi_tilde + db1 * phi_mid + db2 * state1.phi_tilde
        dpi_dt = db0 * state0.pi_tilde + db1 * pi_mid + db2 * state1.pi_tilde

        state_t = BulkState(phi_tilde=phi_t, pi_tilde=pi_t)
        velocity_t = BulkState(phi_tilde=dphi_dt, pi_tilde=dpi_dt)

        return state_t, velocity_t


# =============================================================================
# Path Factory
# =============================================================================


def create_path(
    path_type: PathType,
    geometry=None,
    r_ir: float = 0.0,
    r_uv: float = 1.0,
) -> nn.Module:
    """
    Factory function to create path interpolation module.

    Args:
        path_type: Type of path to create
        geometry: AdS geometry (required for Hermite path)
        r_ir: IR boundary radius
        r_uv: UV boundary radius

    Returns:
        Path module with state_and_target_velocity method

    Raises:
        ValueError: If geometry is required but not provided

    Example
    -------
    >>> path = create_path(PathType.LINEAR)
    >>> # or
    >>> path = create_path(PathType.ADS_AFFINE_HERMITE, geometry=geometry)
    """
    if path_type == PathType.LINEAR:

        class LinearPath(nn.Module):
            def state_and_target_velocity(
                self, s0: BulkState, s1: BulkState, t: Tensor
            ) -> Tuple[BulkState, BulkState]:
                return linear_path_and_velocity(s0, s1, t)

        return LinearPath()

    elif path_type == PathType.ADS_AFFINE_HERMITE:
        if geometry is None:
            raise ValueError("Hermite path requires geometry argument")
        return AdSAffineHermitePath(geometry, r_ir, r_uv)

    elif path_type == PathType.QUADRATIC:
        return QuadraticPath()

    else:
        raise ValueError(f"Unknown path type: {path_type}")
