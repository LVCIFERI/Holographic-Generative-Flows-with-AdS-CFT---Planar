"""
encoding_base.py

Base Encoder Classes and Helper Functions for AdS/CFT Flow Matching
====================================================================

This module provides the foundational components for encoding boundary
data into bulk field representations, including:

1. Encoder protocols and abstract base classes
2. BulkField class for UV-stabilization transformations
3. Bessel function helpers for bulk-boundary propagators

Document References
-------------------
Section 5 (UV stabilization):
    The UV-stabilized variables are defined as:
        Φ̃^(i)(r,·) = e^{(d-Δ_i) r} Φ^(i)(r,·)
        Π̃^(i)(r,·) = e^{(d-Δ_i) r} (Π^(i)(r,·) + (d-Δ_i) Φ^(i)(r,·))

    These remove the exponential decay/growth and give O(1) quantities.

Section 3 (Bulk-boundary propagator):
    For planar AdS, the bulk-boundary propagator in Fourier space is:
        K̂_Δ(z, k) ∝ z^(d/2) |k|^ν K_ν(|k|z)
    where ν = Δ - d/2 and K_ν is the modified Bessel function.

Class Hierarchy
---------------
    EncoderProtocol (Protocol for type checking)
    ├── BaseEncoder (Abstract base class)
    │   └── (Concrete encoders in encoding_spectral.py, encoding_image.py)
    └── BulkField (UV stabilization transformations)

Helper Function Categories
--------------------------
    Bessel Functions:
        - _bessel_k_nu: Modified Bessel K_ν for propagators
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

import numpy as np
import torch
import torch.nn as nn

from ads_cft.registry import register_encoder

# Check for scipy availability
try:
    import scipy.special
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

Tensor = torch.Tensor


# =============================================================================
# Encoder Protocol (for type checking)
# =============================================================================


@runtime_checkable
class EncoderProtocol(Protocol):
    """
    Protocol defining the interface for boundary-to-bulk encoders.

    All encoders must implement:
    - forward(): Encode boundary data to bulk field coefficients
    - decode(): Decode bulk field coefficients to boundary data
    - to_position_space(): Convert spectral to position representation

    Attributes:
        n_channels: Number of field channels
        n_modes: Number of spectral modes per dimension
    """

    n_channels: int
    n_modes: int

    def forward(self, points: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode boundary points to bulk field spectral coefficients.

        Args:
            points: (B, d) boundary point coordinates

        Returns:
            Tuple of:
                phi_hat: (B, 2C, K, K) field spectral coefficients
                pi_hat: (B, 2C, K, K) momentum spectral coefficients
        """
        ...

    def decode(self, phi_hat: Tensor) -> Tensor:
        """
        Decode bulk field coefficients to boundary points.

        Args:
            phi_hat: (B, 2C, K, K) spectral coefficients

        Returns:
            points: (B, d) decoded boundary coordinates
        """
        ...

    def to_position_space(self, phi_hat: Tensor, grid_size: int) -> Tensor:
        """
        Convert spectral coefficients to position space field.

        Args:
            phi_hat: (B, 2C, K, K) spectral coefficients
            grid_size: Output grid resolution

        Returns:
            phi: (B, C, H, W) position space field
        """
        ...


# =============================================================================
# Abstract Base Encoder
# =============================================================================


class BaseEncoder(nn.Module, ABC):
    """
    Abstract base class for boundary-to-bulk encoders.

    Provides common functionality and interface for all encoder types.
    Concrete implementations should inherit from this class.

    Attributes
    ----------
    d : int
        Boundary dimension
    n_channels : int
        Number of field channels (one per conformal dimension)
    n_modes : int
        Number of spectral modes per dimension
    deltas : Tensor
        Conformal dimensions Δ_i for each channel
    nus : Tensor
        Bessel orders ν_i = Δ_i - d/2

    Document Reference
    ------------------
    Section 3: The encoder implements the bulk-boundary propagator,
    mapping boundary data x' to bulk field coefficients.
    """

    def __init__(
        self,
        d: int,
        deltas: Sequence[float],
        n_modes: int,
    ) -> None:
        """
        Initialize base encoder.

        Args:
            d: Boundary dimension
            deltas: Conformal dimensions for each channel
            n_modes: Spectral modes per dimension
        """
        super().__init__()
        self.d = d
        self.n_channels = len(deltas)
        self.n_modes = n_modes

        deltas_t = torch.tensor(list(deltas), dtype=torch.float32)
        self.register_buffer("deltas", deltas_t)

        # Bessel orders: ν_i = Δ_i - d/2
        nus = deltas_t - d / 2.0
        self.register_buffer("nus", nus)

    @abstractmethod
    def forward(self, points: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode boundary points to bulk field coefficients.

        Args:
            points: (B, d) boundary coordinates

        Returns:
            Tuple of (phi_hat, pi_hat) spectral coefficients
        """
        pass

    @abstractmethod
    def decode(self, phi_hat: Tensor) -> Tensor:
        """
        Decode bulk coefficients to boundary points.

        Args:
            phi_hat: (B, 2C, K, K) spectral coefficients

        Returns:
            points: (B, d) decoded coordinates
        """
        pass

    @abstractmethod
    def to_position_space(
        self, phi_hat: Tensor, grid_size: int = 64
    ) -> Tensor:
        """
        Convert spectral coefficients to position space.

        Args:
            phi_hat: Spectral coefficients
            grid_size: Output resolution

        Returns:
            Position space field
        """
        pass


# =============================================================================
# BulkField: UV Stabilization (Document Section 5)
# =============================================================================


class BulkField(nn.Module):
    """
    Channelwise bulk scalar specification via UV dimensions Δ_i.

    Implements the UV-stabilization transformation that removes the
    exponential decay/growth of fields near the AdS boundary.

    Document eq (mass-dimension):
        m_i² = Δ_i(Δ_i - d)

    Attributes
    ----------
    d : int
        Boundary dimension
    deltas : Tensor
        Conformal dimensions Δ_i
    m2 : Tensor
        Bulk masses m_i² = Δ_i(Δ_i - d)
    n_fields : int
        Number of field channels

    Example
    -------
    >>> bulk_field = BulkField(d=2, deltas=[1.5, 2.0])
    >>> phi_tilde, pi_tilde = bulk_field.stabilize(phi, pi, r)
    >>> phi, pi = bulk_field.destabilize(phi_tilde, pi_tilde, r)
    """

    def __init__(self, d: int, deltas: Sequence[float]) -> None:
        """
        Initialize BulkField.

        Args:
            d: Boundary dimension (must be positive)
            deltas: Conformal dimensions (must be non-empty)

        Raises:
            ValueError: If d <= 0 or deltas is empty
        """
        super().__init__()
        if d <= 0:
            raise ValueError(f"d must be positive, got {d}")
        if len(deltas) == 0:
            raise ValueError("deltas must be non-empty")

        self.d = int(d)
        deltas_t = torch.tensor(list(deltas), dtype=torch.float32)
        self.register_buffer("deltas", deltas_t)
        self.register_buffer("m2", deltas_t * (deltas_t - float(d)))
        self.register_buffer("alpha", 2.0 * deltas_t - float(d))

    @property
    def n_fields(self) -> int:
        """Number of field channels."""
        return int(self.deltas.numel())

    def _broadcast_deltas(self, x: Tensor) -> Tensor:
        """
        Broadcast deltas to shape (1, C, 1, 1, ...) matching x.

        Args:
            x: Input tensor with shape (B, C, ...)

        Returns:
            Deltas broadcast to match x dimensions

        Raises:
            ValueError: If x has wrong shape or channel mismatch
        """
        if x.ndim < 2:
            raise ValueError("Expected x to have shape (B, C, ...)")
        if x.shape[1] != self.n_fields:
            raise ValueError(
                f"Channel mismatch: x has C={x.shape[1]}, "
                f"BulkField has {self.n_fields}"
            )
        shape = [1, self.n_fields] + [1] * (x.ndim - 2)
        return self.deltas.to(device=x.device, dtype=x.dtype).view(*shape)

    def _broadcast_r(self, r: Tensor, target_ndim: int) -> Tensor:
        """
        Broadcast r to shape (B, 1, 1, ...) matching target_ndim.

        Args:
            r: Radial coordinate (scalar, (B,), or higher)
            target_ndim: Target number of dimensions

        Returns:
            r broadcast to target dimensions
        """
        if r.ndim == 0:
            r = r.view(1, 1, *([1] * (target_ndim - 2)))
        elif r.ndim == 1:
            r = r.view(-1, 1, *([1] * (target_ndim - 2)))
        else:
            while r.ndim < target_ndim:
                r = r.unsqueeze(-1)
        return r

    def stabilize(
        self, phi: Tensor, pi: Tensor, r: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Map physical variables (Φ, Π) to UV-stabilized variables (Φ̃, Π̃).

        Document eq (uv-stable-def):
            Φ̃^(i)(r,·) = e^{(d-Δ_i) r} Φ^(i)(r,·)
            Π̃^(i)(r,·) = e^{(d-Δ_i) r} (Π^(i)(r,·) + (d-Δ_i) Φ^(i)(r,·))

        The UV-stabilized variables are O(1) near the boundary (r → ∞)
        rather than decaying exponentially.

        Args:
            phi: (B, C, *spatial) physical field Φ
            pi: (B, C, *spatial) physical momentum Π
            r: Radial coordinate

        Returns:
            Tuple of (phi_tilde, pi_tilde) UV-stabilized variables
        """
        deltas = self._broadcast_deltas(phi)
        r_b = self._broadcast_r(
            r.to(device=phi.device, dtype=phi.dtype), phi.ndim
        )

        scale = torch.exp((self.d - deltas) * r_b)
        phi_tilde = scale * phi
        pi_tilde = scale * (pi + (self.d - deltas) * phi)
        return phi_tilde, pi_tilde

    def destabilize(
        self, phi_tilde: Tensor, pi_tilde: Tensor, r: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Inverse: map (Φ̃, Π̃) to physical (Φ, Π).

        Document eq (uv-stable-inverse):
            Φ^(i)(r,·) = e^{-(d-Δ_i) r} Φ̃^(i)(r,·)
            Π^(i)(r,·) = e^{-(d-Δ_i) r} (Π̃^(i)(r,·) - (d-Δ_i) Φ̃^(i)(r,·))

        Args:
            phi_tilde: (B, C, *spatial) UV-stabilized field Φ̃
            pi_tilde: (B, C, *spatial) UV-stabilized momentum Π̃
            r: Radial coordinate

        Returns:
            Tuple of (phi, pi) physical variables
        """
        deltas = self._broadcast_deltas(phi_tilde)
        r_b = self._broadcast_r(
            r.to(device=phi_tilde.device, dtype=phi_tilde.dtype), phi_tilde.ndim
        )

        inv_scale = torch.exp(-(self.d - deltas) * r_b)
        phi = inv_scale * phi_tilde
        pi = inv_scale * (pi_tilde - (self.d - deltas) * phi_tilde)
        return phi, pi


# =============================================================================
# Bessel Function Helpers
# =============================================================================


def _bessel_k_nu(nu: float, x: Tensor, eps: float = 1e-10) -> Tensor:
    """
    Compute modified Bessel function of the second kind K_ν(x).

    Uses different methods depending on ν:
    - ν = 0: torch.special.modified_bessel_k0 (if available)
    - ν = 1: torch.special.modified_bessel_k1 (if available)
    - ν = 0.5: Exact formula K_{0.5}(x) = sqrt(π/(2x)) e^{-x}
    - ν = 1.5: Exact formula K_{1.5}(x) = sqrt(π/(2x)) e^{-x} (1 + 1/x)
    - ν = 2.5: Exact formula K_{2.5}(x) = sqrt(π/(2x)) e^{-x} (1 + 3/x + 3/x^2)
    - General ν: Uniform asymptotic expansion

    For numerical stability, returns 0 for very large x and uses
    asymptotic forms for small x.

    Args:
        nu: Bessel order ν
        x: Argument tensor
        eps: Small epsilon for numerical stability

    Returns:
        K_ν(x) values

    Document Reference:
        Section 3: The bulk-boundary propagator involves K_ν(|k|z)
        where ν = Δ - d/2.
    """
    x_safe = x.clamp(min=eps)

    # Check for special cases with exact formulas
    if abs(nu - 0.5) < 1e-6:
        # K_{0.5}(x) = sqrt(π/(2x)) * e^{-x}
        return torch.sqrt(math.pi / (2.0 * x_safe)) * torch.exp(-x_safe)

    elif abs(nu - 1.5) < 1e-6:
        # K_{1.5}(x) = sqrt(π/(2x)) * e^{-x} * (1 + 1/x)
        return (
            torch.sqrt(math.pi / (2.0 * x_safe))
            * torch.exp(-x_safe)
            * (1.0 + 1.0 / x_safe)
        )

    elif abs(nu - 2.5) < 1e-6:
        # K_{2.5}(x) = sqrt(π/(2x)) * e^{-x} * (1 + 3/x + 3/x²)
        inv_x = 1.0 / x_safe
        return (
            torch.sqrt(math.pi / (2.0 * x_safe))
            * torch.exp(-x_safe)
            * (1.0 + 3.0 * inv_x + 3.0 * inv_x * inv_x)
        )

    elif abs(nu) < 1e-6:
        # K_0(x) - use PyTorch's built-in if available
        if hasattr(torch.special, "modified_bessel_k0"):
            return torch.special.modified_bessel_k0(x_safe)
        else:
            # Approximation for K_0: sqrt(π/(2x))e^{-x} for moderate/large x
            return torch.sqrt(math.pi / (2.0 * x_safe)) * torch.exp(-x_safe)

    elif abs(nu - 1.0) < 1e-6:
        # K_1(x) - use PyTorch's built-in if available
        if hasattr(torch.special, "modified_bessel_k1"):
            return torch.special.modified_bessel_k1(x_safe)
        else:
            # K_1(x) ≈ sqrt(π/(2x))e^{-x}(1 + 3/(8x)) for moderate/large x
            return (
                torch.sqrt(math.pi / (2.0 * x_safe))
                * torch.exp(-x_safe)
                * (1.0 + 0.375 / x_safe)
            )

    else:
        # General case - use uniform asymptotic expansion
        # K_ν(x) ≈ sqrt(π/(2x)) * e^{-x} * (1 + (4ν²-1)/(8x) + (16ν^4 - 40ν² + 9)/(128x^2) + ...)
        sqrt_term = torch.sqrt(math.pi / (2.0 * x_safe))
        exp_term = torch.exp(-x_safe)

        # Correction up to 1/x^3 term
        nu_sq = nu**2
        nu_4 = nu**4
        nu_6 = nu**6

        correction_1 = 1.0 + (4.0 * nu_sq - 1.0) / (8.0 * x_safe)
        correction_2 = (16.0 * nu_4 - 40 * nu_sq + 9)/(128 * x_safe**2)
        correction_3 = (64.0 * nu_6 - 560 * nu_4 + 1036 * nu_sq - 225)/(3072 * x_safe**3)

        result = sqrt_term * exp_term * (correction_1 + correction_2 + correction_3)

        # For small x, use the power-law behavior K_ν(x) ~ (Γ(ν)/2)(2/x)^ν
        small_x_mask = x_safe < 0.7
        if small_x_mask.any() and nu > 0:
            gamma_nu = math.gamma(nu)
            small_x_approx = 0.5 * gamma_nu * torch.pow(2.0 / x_safe, nu)
            result = torch.where(small_x_mask, small_x_approx, result)

        return result.clamp(max=1e10)



# =============================================================================
# Utility Functions
# =============================================================================


def compute_spectral_envelope_planar(
    k_mag: Tensor,
    delta: float,
    r_uv: float,
    regularization: float = 1e-2,
) -> Tuple[Tensor, Tensor]:
    """
    Compute Bessel-based spectral envelope for planar geometry.

    The UV-stabilized bulk-boundary propagator in Fourier space is:
        κ̃_{|k|}(r) = (2/Γ(ν)) (e^{-r}|k|/2)^ν K_ν(|k|e^{-r})

    where ν = Δ - d/2 and ξ = |k|e^{-r} = |k|z.

    This comes from the definition K̃ = e^{(d-Δ)r} K applied to the unstabilized
    propagator κ_{|k|}(r) = (2/Γ(ν)) e^{-dr/2} (|k|/2)^ν K_ν(ξ).

    Args:
        k_mag: (K, K) tensor of |k| values
        delta: Conformal dimension Δ
        r_uv: UV radius
        regularization: Small epsilon for stability

    Returns:
        Tuple of (envelope_phi, envelope_pi) tensors

    Document Reference:
        Section 2.4: UV-stabilized propagator mode coefficients
    """
    d = 2  # Assume 2D boundary
    nu = delta - d / 2.0
    z_uv = math.exp(-r_uv)

    # xi = |k| * z = |k| * e^{-r} is the argument for Bessel functions
    xi = k_mag * z_uv
    K_nu = _bessel_k_nu(nu, xi, eps=regularization)

    # UV-stabilized propagator: K̃ = e^{(d-Δ)r} K
    # κ̃_{|k|}(r) = (2/Γ(ν)) (ξ/2)^ν K_ν(ξ)
    xi_power = torch.pow(xi, nu)
    envelope_phi_raw =  xi_power * K_nu

    # Normalize so zero-mode has unit amplitude
    norm_val = envelope_phi_raw[0, 0].item()
    if abs(norm_val) > 1e-10:
        envelope_phi = envelope_phi_raw / norm_val
    else:
        envelope_phi = envelope_phi_raw

    # Momentum envelope: Π̃ = ∂_r Φ̃
    # ∂_r[ξ^ν K_ν(ξ)] = ξ^{ν+1} K_{ν-1}(ξ)
    K_nu_minus_1 = _bessel_k_nu(abs(nu - 1.0), xi, eps=regularization)
    envelope_pi_raw = xi_power * (xi * K_nu_minus_1)

    if abs(norm_val) > 1e-10:
        envelope_pi = envelope_pi_raw / norm_val
    else:
        envelope_pi = envelope_pi_raw

    return envelope_phi, envelope_pi
