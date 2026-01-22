"""
laplacian_base.py

Slice Laplacian Operators for AdS/CFT Flow Matching Framework
==============================================================

This module provides the base class and diagonal implementations for the
slice Laplacian operator (-Δ̂_g) appearing in the Klein-Gordon equation.

Paper References
----------------
Paper Section 2.2 (Spectral decomposition of Klein-Gordon):
    The Laplacian eigenfunctions satisfy:
        Δ̂_g Y_λ^α(x) = -λ Y_λ^α(x)

    For planar (flat) slices with periodic boundary conditions,
    the eigenbasis is Fourier modes with eigenvalues λ = |k|².

Paper eq. 3 (Klein-Gordon equation):
    (∂_r² + d·f'/f·∂_r + f⁻²Δ̂_g - m²)Φ = 0

    The Laplacian term f⁻²Δ̂_g couples the radial evolution to
    the spatial structure of the field.

Paper eqs. 20-21 (UV-stabilized, planar):
    dπ̃_k/dr = |k|² e^{-2r} φ̃_k - (2Δ - d)π̃_k

    Here the Laplacian eigenvalue |k|² appears explicitly.

Laplacian Implementation
------------------------
This module provides DIAGONAL implementations where the Laplacian is
represented in an eigenbasis:

1. PlanarSpectralLaplacian: FFT-based for periodic box [0, L)^d
   - Eigenvalues: λ(k) = |k|² = (2π)² Σᵢ (nᵢ/Lᵢ)²
   - Fast: O(N log N) via FFT

2. SpectralLaplacian: Generic diagonal Laplacian
   - Works with any precomputed eigenvalue grid

Class Hierarchy
---------------
    SliceLaplacian (abstract base, nn.Module)
    ├── PlanarSpectralLaplacian
    └── SpectralLaplacian

Factory Functions
-----------------
    build_slice_laplacian: Create appropriate Laplacian from geometry + discretization
    slice_quadrature_weights: Compute L² quadrature weights
    sample_isotropic_noise_from_quadrature: Sample L²-isotropic Gaussian noise
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from ads_cft.config import DiscretizationConfig, SliceGeometry
from ads_cft.registry import register_laplacian

Tensor = torch.Tensor


# =============================================================================
# Abstract Base Class
# =============================================================================


class SliceLaplacian(nn.Module, ABC):
    """
    Abstract base class for slice Laplacian operators.

    Implements the interface for applying the positive semidefinite
    operator (-Δ_ĝ) on the slice manifold Σ.

    Document convention (eq lap-eigs):
        -Δ_ĝ ψ_j = λ_j ψ_j, λ_j ≥ 0

    All implementations must provide:
        - apply_minus_laplacian(x): Apply (-Δ_ĝ) to field x
        - diag_eigs(device, dtype): Return eigenvalues if diagonal, else None

    The Laplacian appears in the UV-stabilized backbone:
        ∂_r Π̃ = ... - (1/f²)Δ_ĝ Φ̃ + ...
    """

    @abstractmethod
    def apply_minus_laplacian(self, x: Tensor) -> Tensor:
        """
        Apply (-Δ_ĝ) to field x.

        Args:
            x: Field tensor, shape depends on implementation
               Typically (B, C, *spatial_dims)

        Returns:
            (-Δ_ĝ)x with same shape as input
        """
        pass

    def diag_eigs(
        self, device: torch.device, dtype: torch.dtype
    ) -> Optional[Tensor]:
        """
        Return eigenvalues λ_j if the basis diagonalizes (-Δ_ĝ).

        Args:
            device: Device for output tensor
            dtype: Data type for output tensor

        Returns:
            Eigenvalue tensor if diagonal representation, None otherwise
        """
        return None


# =============================================================================
# Planar Spectral Laplacian (FFT-based)
# =============================================================================


@register_laplacian("planar_spectral")
class PlanarSpectralLaplacian(SliceLaplacian):
    """
    Spectral Laplacian on periodic box [0, L₁) × ... × [0, L_d).

    Document Section 2.3.1:
        λ(k) = ||k||², where k = (2π/L_i) ν_i, ν_i ∈ {-n_i/2, ..., n_i/2 - 1}

    In physical representation, the field is on a grid of shape (n₁, ..., n_d).
    Apply (-Δ_ĝ) via FFT: F^{-1}[ λ(k) F[x] ]

    Parameters
    ----------
    grid_shape : Tuple[int, ...]
        Number of grid points in each dimension, e.g., (64, 64) for 2D
    box_lengths : Tuple[float, ...]
        Physical domain size in each dimension, e.g., (8.0, 8.0)
    device : torch.device
        Computation device
    dtype : torch.dtype
        Data type for computations

    Example
    -------
    >>> laplacian = PlanarSpectralLaplacian(
    ...     grid_shape=(64, 64),
    ...     box_lengths=(8.0, 8.0),
    ...     device=torch.device('cuda'),
    ... )
    >>> phi = torch.randn(16, 2, 64, 64, device='cuda')  # (B, C, H, W)
    >>> lap_phi = laplacian.apply_minus_laplacian(phi)   # (-Δ)φ
    """

    def __init__(
        self,
        grid_shape: Tuple[int, ...],
        box_lengths: Tuple[float, ...],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initialize planar spectral Laplacian.

        Args:
            grid_shape: Spatial grid dimensions
            box_lengths: Physical domain size per dimension
            device: Computation device
            dtype: Data type

        Raises:
            ValueError: If grid_shape and box_lengths have different lengths
        """
        super().__init__()
        if len(grid_shape) != len(box_lengths):
            raise ValueError(
                f"grid_shape and box_lengths must have same length, "
                f"got {len(grid_shape)} and {len(box_lengths)}"
            )

        self.grid_shape = grid_shape
        self.box_lengths = box_lengths
        self.ndim = len(grid_shape)

        # Compute squared frequencies for each dimension
        # For dimension i with n_i points and length L_i:
        # k_i = (2π/L_i) * fftfreq(n_i) * n_i = 2π/L_i * [-n/2, ..., n/2-1]
        freq_grids = []
        for i, (n, L) in enumerate(zip(grid_shape, box_lengths)):
            freq = torch.fft.fftfreq(
                n, d=L / (2.0 * math.pi * n), device=device, dtype=dtype
            )
            freq_grids.append(freq)

        # Build full k² grid via outer sum
        k_sq = torch.zeros(grid_shape, device=device, dtype=dtype)
        for i, freq in enumerate(freq_grids):
            shape = [1] * self.ndim
            shape[i] = len(freq)
            k_sq = k_sq + freq.view(*shape) ** 2

        self.register_buffer("k_sq", k_sq)

    def apply_minus_laplacian(self, x: Tensor) -> Tensor:
        """
        Apply (-Δ_ĝ) via FFT.

        Args:
            x: Field tensor of shape (B, C, *grid_shape)

        Returns:
            (-Δ)x with same shape

        Implementation:
            1. FFT over spatial dimensions
            2. Multiply by k² eigenvalues
            3. Inverse FFT
        """
        # FFT over spatial dimensions (all dims after batch and channel)
        spatial_dims = tuple(range(2, x.ndim))
        x_hat = torch.fft.fftn(x, dim=spatial_dims)

        # Multiply by k²
        result_hat = x_hat * self.k_sq

        # Inverse FFT
        result = torch.fft.ifftn(result_hat, dim=spatial_dims).real
        return result

    def diag_eigs(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """
        Return eigenvalues λ(k) = ||k||².

        Args:
            device: Output device
            dtype: Output dtype

        Returns:
            Eigenvalue grid of shape grid_shape
        """
        return self.k_sq.to(device=device, dtype=dtype)

# =============================================================================
# Generic Spectral Laplacian (Diagonal)
# =============================================================================


@register_laplacian("diagonal")
class SpectralLaplacian(nn.Module):
    """
    Generic diagonal Laplacian operator in spectral space.

    This is the simplest Laplacian: Δ φ̂_k = -|k|² φ̂_k

    Works with any precomputed eigenvalue grid (k² values).
    No FFT needed - just multiply by precomputed -k²!

    Parameters
    ----------
    n_modes : int
        Number of modes per dimension (for 2D grid)
    n_channels : int
        Number of field channels
    k_sq : Tensor
        Precomputed |k|² grid, shape (K, K) for 2D

    Attributes
    ----------
    minus_k_sq : Tensor
        Stored as -k² for convenience

    Example
    -------
    >>> k_sq = compute_k_squared(n_modes=16, domain_size=8.0)
    >>> laplacian = SpectralLaplacian(n_modes=16, n_channels=2, k_sq=k_sq)
    >>> phi_hat = torch.randn(16, 4, 16, 16)  # (B, 2C, K, K)
    >>> lap_phi_hat = laplacian.apply_minus_laplacian(phi_hat)
    """

    def __init__(
        self, n_modes: int, n_channels: int, k_sq: Tensor
    ) -> None:
        """
        Initialize spectral Laplacian.

        Args:
            n_modes: Modes per dimension
            n_channels: Field channels
            k_sq: |k|² eigenvalue grid
        """
        super().__init__()
        self.n_modes = n_modes
        self.n_channels = n_channels
        # Store -k² for efficiency
        self.register_buffer("minus_k_sq", -k_sq)

    def apply_minus_laplacian(self, phi_hat: Tensor) -> Tensor:
        """
        Apply (-Δ) in spectral space.

        Args:
            phi_hat: (B, 2C, K, K) spectral coefficients

        Returns:
            lap_phi_hat: (B, 2C, K, K) = |k|² φ̂

        Note:
            -Δ φ = |k|² φ in Fourier space, so (-Δ) φ̂ = |k|² φ̂
        """
        k_sq = (-self.minus_k_sq).to(device=phi_hat.device, dtype=phi_hat.dtype)
        return phi_hat * k_sq.unsqueeze(0).unsqueeze(0)  # (B, 2C, K, K)

    def diag_eigs(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Return eigenvalues |k|²."""
        return (-self.minus_k_sq).to(device=device, dtype=dtype)


# =============================================================================
# Factory Functions
# =============================================================================


def build_slice_laplacian(
    geometry: "AdSGeometry",
    disc: DiscretizationConfig,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Optional[SliceLaplacian]:
    """
    Build appropriate slice Laplacian for geometry and discretization.

    Factory function that creates the correct Laplacian implementation
    based on the slice geometry and discretization configuration.

    Args:
        geometry: AdS geometry instance
        disc: Discretization configuration
        device: Computation device
        dtype: Data type

    Returns:
        SliceLaplacian instance, or None if configuration is incomplete

    Example:
    -------
    >>> from ads_cft.geometry import make_geometry
    >>> from ads_cft.config import GeometryConfig, DiscretizationConfig
    >>> 
    >>> geo_cfg = GeometryConfig(d=2, slice_geometry=SliceGeometry.PLANAR)
    >>> disc_cfg = DiscretizationConfig(grid_shape=(64, 64), box_lengths=(8.0, 8.0))
    >>> geometry = make_geometry(geo_cfg)
    >>> laplacian = build_slice_laplacian(geometry, disc_cfg)
    """
    if geometry.slice_geometry == SliceGeometry.PLANAR:
        if disc.grid_shape is not None and disc.box_lengths is not None:
            return PlanarSpectralLaplacian(
                grid_shape=disc.grid_shape,
                box_lengths=disc.box_lengths,
                device=device,
                dtype=dtype,
            )

    elif geometry.slice_geometry == SliceGeometry.FLAT:
        # ABLATION: Same slice Laplacian as planar (flat ℝ^d)
        if disc.grid_shape is not None and disc.box_lengths is not None:
            return PlanarSpectralLaplacian(
                grid_shape=disc.grid_shape,
                box_lengths=disc.box_lengths,
                device=device,
                dtype=dtype,
            )

    # No Laplacian for point data or unsupported configs
    return None


def slice_quadrature_weights(
    geometry: "AdSGeometry",
    disc: DiscretizationConfig,
    r: float,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Optional[Tensor]:
    """
    Compute quadrature weights for L² inner product at radius r.

    The L² norm on the slice is:
        ||φ||²_{g_r} = ∫_Σ |φ|² dvol_{g_r} = ∫_Σ |φ|² ω(r) dvol_ĝ

    This returns the quadrature weights for numerical integration.

    Args:
        geometry: AdS geometry instance
        disc: Discretization configuration
        r: Radial coordinate
        device: Computation device
        dtype: Data type

    Returns:
        Quadrature weight tensor, or None if configuration is incomplete

    Note:
        The weights include the slice volume factor ω(r).
    """
    omega = geometry.slice_weight(torch.tensor(r, device=device, dtype=dtype))

    if geometry.slice_geometry == SliceGeometry.PLANAR:
        if disc.grid_shape is not None and disc.box_lengths is not None:
            # Uniform weights: dx₁ × ... × dx_d × ω(r)
            vol_per_cell = 1.0
            for n, L in zip(disc.grid_shape, disc.box_lengths):
                vol_per_cell *= L / n
            return omega * vol_per_cell * torch.ones(
                disc.grid_shape, device=device, dtype=dtype
            )

    elif geometry.slice_geometry == SliceGeometry.FLAT:
        # ABLATION: Same quadrature as planar, but ω(r)=1 for flat
        if disc.grid_shape is not None and disc.box_lengths is not None:
            vol_per_cell = 1.0
            for n, L in zip(disc.grid_shape, disc.box_lengths):
                vol_per_cell *= L / n
            return omega * vol_per_cell * torch.ones(
                disc.grid_shape, device=device, dtype=dtype
            )

    # For other geometries, would need specific quadrature rules
    return None


def sample_isotropic_noise_from_quadrature(
    weights: Tensor,
    shape: Tuple[int, ...],
    sigma: float,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """
    Sample L²-isotropic Gaussian noise given quadrature weights.

    For the noise to be isotropic in the L²(Σ, dvol_{g_r}) inner product,
    we scale the standard Gaussian noise by the inverse square root of
    the quadrature weights:

        ξ_p = σ η_p / √w_p,  where η_p ~ N(0, 1)

    This ensures:
        E[∫ |ξ|² dvol_{g_r}] = σ²

    Args:
        weights: Quadrature weights for integration
        shape: Output tensor shape
        sigma: Target standard deviation
        generator: Optional random number generator

    Returns:
        L²-isotropic noise tensor

    Document Reference:
        Section 5.1 (eq lift-identity): The noise ξ should be
        L²(Σ, dvol_{g_{r_UV}})-isotropic Gaussian.
    """
    eta = torch.randn(
        shape, device=weights.device, dtype=weights.dtype, generator=generator
    )
    # Broadcast weights to match shape
    w = weights
    while w.ndim < eta.ndim:
        w = w.unsqueeze(0)
    return sigma * eta / torch.sqrt(w + 1e-10)


# =============================================================================
# Type Annotations for Forward References
# =============================================================================

# Import at runtime to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ads_cft.geometry import AdSGeometry
