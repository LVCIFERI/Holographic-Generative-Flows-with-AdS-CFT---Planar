"""
geometry.py

AdS Geometry Classes for the Flow Matching Framework
====================================================

Implementation of AdS foliations and warp factors following the paper
"Holographic generative flows with AdS/CFT".

Paper Section 2: AdS foliations and warp factors
------------------------------------------------
Single-warp AdS foliations (warped product, paper eq. 1):
    ds² = dr² + f(r)² ĝ_ab dx^a dx^b

    - PLANAR:     Σ ≅ ℝ^d,  f(r) = e^r,      f'/f = 1

Core geometric quantities (implementation-critical):
----------------------------------------------------
- Slice volume weight ω(r) such that dvol_{g_r} = ω(r) dvol_ĝ
    single-warp: ω(r) = f(r)^d

- Radial friction κ(r) used in the first-order KG system (paper eq. 3):
    single-warp: κ(r) = d (f'/f)

- Affine parameter u(r) for AdS-geometric flow-matching paths (paper eq. 29):
    For planar AdS: u(r) = [1 - e^{-d(r - r_IR)}] / [1 - e^{-d(r_UV - r_IR)}]
    General: u(r) = (1/Z) ∫_{r_IR}^r 1/ω(s) ds

Class Hierarchy
---------------
    AdSGeometry (abstract base)
    ├── PlanarAdS                 (f = e^r)
    ├── FlatGeometry              (f = 1, ablation)
    └── HyperscalingViolatingAdS  (f = [(1-p)r]^{-p/(1-p)})

HSV Convention Note
-------------------
The code uses p ∈ [0, 1) where:
    p = 0  → flat space (f = 1)
    p → 1  → planar AdS (f → e^r)

This is INVERTED from the paper's convention where p=0 is AdS and p=1 is flat.
The mathematical formulas are adjusted accordingly.

Adding New Geometries
---------------------
To add a new geometry:

1. Create a subclass of AdSGeometry
2. Implement the abstract methods (f, log_f_prime for single-warp)
3. Register with @register_geometry("name")

Example:
    >>> from ads_cft.registry import register_geometry
    >>> 
    >>> @register_geometry("de_sitter")
    >>> class DeSitterGeometry(AdSGeometry):
    ...     def f(self, r: Tensor) -> Tensor:
    ...         return torch.cos(r)  # dS has f = cos(r)
    ...     
    ...     def log_f_prime(self, r: Tensor) -> Tensor:
    ...         return -torch.tan(r)  # f'/f = -tan(r)
    ...     
    ...     # ... multi-warp stubs ...
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch

from ads_cft.config import GeometryConfig, SliceGeometry
from ads_cft.registry import register_geometry

Tensor = torch.Tensor


# =============================================================================
# Helper Functions
# =============================================================================


def _as_tensor(
    x: Union[float, Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Convert scalar or tensor to tensor on specified device/dtype.

    Args:
        x: Input value (scalar or tensor)
        device: Target device
        dtype: Target dtype

    Returns:
        Tensor on specified device with specified dtype
    """
    if isinstance(x, Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


# =============================================================================
# Base Geometry Class
# =============================================================================


class AdSGeometry(ABC):
    """
    Abstract base class for AdS warped-product geometries.

    Concrete subclasses implement:
      - Single-warp: f(r), log_f_prime(r) = (f'/f)(r)
      - Multi-warp: a(r), b(r), log_a_prime(r), log_b_prime(r)

    This base provides: ω(r), κ(r), affine parameter utilities, quadrature helpers.

    Document references:
        - Section 2: Geometry definitions
        - Section 2.1: Geometry details for all slicings
        - Section 9: Affine parameter for flow matching paths

    Parameters
    ----------
    cfg : GeometryConfig
        Geometry configuration containing d, slice_geometry, etc.

    Attributes
    ----------
    cfg : GeometryConfig
        Stored configuration
    d : int
        Boundary/slice dimension
    """

    def __init__(self, cfg: GeometryConfig) -> None:
        """
        Initialize geometry from configuration.

        Args:
            cfg: Geometry configuration

        Raises:
            ValueError: If d <= 0
        """
        if cfg.d <= 0:
            raise ValueError(f"Boundary dimension d must be >= 1, got d={cfg.d}")
        self.cfg = cfg
        self.d = int(cfg.d)
        self._affine_cache: Dict[
            Tuple[float, float, torch.device, torch.dtype],
            Dict[str, Tensor]
        ] = {}

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def device(self) -> torch.device:
        """Computation device."""
        return torch.device(self.cfg.device)

    @property
    def dtype(self) -> torch.dtype:
        """Computation dtype."""
        return self.cfg.dtype

    @property
    def slice_geometry(self) -> SliceGeometry:
        """Slice geometry type."""
        return self.cfg.slice_geometry

    @property
    def is_multiwarp(self) -> bool:
        """True for multiply-warped geometries (not supported in planar-only version)."""
        return False

    @property
    def is_hsv(self) -> bool:
        """
        True for hyperscaling-violating geometries.
        
        HSV geometries require special handling in Hermite paths because
        ω(r) decreases toward UV (opposite of standard AdS), making the
        affine-parameter formulation numerically unstable.
        
        Subclasses override this to return True.
        """
        return False

    # =========================================================================
    # Abstract Methods: Single-warp
    # =========================================================================

    @abstractmethod
    def f(self, r: Tensor) -> Tensor:
        """
        Warp factor f(r) for single-warp metrics.

        Document eq (warped-metric): ds² = dr² + f(r)² ĝ

        Args:
            r: Radial coordinate(s)

        Returns:
            Warp factor value(s)
        """
        pass

    @abstractmethod
    def log_f_prime(self, r: Tensor) -> Tensor:
        """
        Return (f'/f)(r) for single-warp geometries.

        This is the logarithmic derivative of the warp factor.

        Args:
            r: Radial coordinate(s)

        Returns:
            (f'/f) value(s)
        """
        pass

    # =========================================================================
    # Abstract Methods: Multi-warp
    # =========================================================================

    @abstractmethod
    def a(self, r: Tensor) -> Tensor:
        """
        Warp factor a(r) for τ direction in multiply-warped metrics.

        Document Section 2.2: ds² = dr² + a(r)² dτ² + b(r)² dΩ²

        Args:
            r: Radial coordinate(s)

        Returns:
            a(r) value(s)
        """
        pass

    @abstractmethod
    def b(self, r: Tensor) -> Tensor:
        """
        Warp factor b(r) for sphere direction in multiply-warped metrics.

        Args:
            r: Radial coordinate(s)

        Returns:
            b(r) value(s)
        """
        pass

    @abstractmethod
    def log_a_prime(self, r: Tensor) -> Tensor:
        """
        Return (a'/a)(r) for multi-warp geometries.

        Args:
            r: Radial coordinate(s)

        Returns:
            (a'/a) value(s)
        """
        pass

    @abstractmethod
    def log_b_prime(self, r: Tensor) -> Tensor:
        """
        Return (b'/b)(r) for multi-warp geometries.

        Args:
            r: Radial coordinate(s)

        Returns:
            (b'/b) value(s)
        """
        pass

    # =========================================================================
    # Derived Geometric Scalars (Document Section 2.1)
    # =========================================================================

    def slice_weight(self, r: Tensor) -> Tensor:
        """
        Volume weight ω(r) such that dvol_{g_r} = ω(r) dvol_ĝ.

        Document eq (slice-volume):
            - Single-warp: ω(r) = f(r)^d
            - Multi-warp:  ω(r) = a(r) b(r)^{d-1}

        Args:
            r: Radial coordinate(s)

        Returns:
            Volume weight ω(r)
        """
        r = _as_tensor(r, device=self.device, dtype=self.dtype)
        if self.is_multiwarp:
            return self.a(r) * (self.b(r) ** (self.d - 1))
        return self.f(r) ** self.d

    def kappa(self, r: Tensor) -> Tensor:
        """
        Radial friction κ(r) appearing in the first-order KG system.

        Document specification:
            - Single-warp: κ(r) = d (f'/f)
            - Multi-warp:  κ(r) = (a'/a) + (d-1)(b'/b)

        Args:
            r: Radial coordinate(s)

        Returns:
            Friction coefficient κ(r)
        """
        r = _as_tensor(r, device=self.device, dtype=self.dtype)
        if self.is_multiwarp:
            return self.log_a_prime(r) + (self.d - 1) * self.log_b_prime(r)
        return float(self.d) * self.log_f_prime(r)

    def slice_weight_prime(self, r: Tensor) -> Tensor:
        """
        Derivative ω'(r) = dω/dr.

        Uses identity: ω'/ω = κ, so ω' = ω κ.

        Args:
            r: Radial coordinate(s)

        Returns:
            dω/dr
        """
        r = _as_tensor(r, device=self.device, dtype=self.dtype)
        return self.slice_weight(r) * self.kappa(r)

    def f_inv2(self, r: Tensor) -> Tensor:
        """
        Return 1/f(r)² for single-warp geometries.

        Used in the Laplacian term of the KG equation.

        Args:
            r: Radial coordinate(s)

        Returns:
            1/f(r)²
        """
        r = _as_tensor(r, device=self.device, dtype=self.dtype)
        f_val = self.f(r)
        return 1.0 / (f_val * f_val)

    def a_inv2(self, r: Tensor) -> Tensor:
        """
        Return 1/a(r)² for multi-warp geometries.

        Args:
            r: Radial coordinate(s)

        Returns:
            1/a(r)²
        """
        r = _as_tensor(r, device=self.device, dtype=self.dtype)
        a_val = self.a(r)
        return 1.0 / (a_val * a_val)

    def b_inv2(self, r: Tensor) -> Tensor:
        """
        Return 1/b(r)² for multi-warp geometries.

        Args:
            r: Radial coordinate(s)

        Returns:
            1/b(r)²
        """
        r = _as_tensor(r, device=self.device, dtype=self.dtype)
        b_val = self.b(r)
        return 1.0 / (b_val * b_val)

    # =========================================================================
    # Affine Parameter u(r) for AdS-geometric Flow Matching
    # (Document Section 9)
    # =========================================================================

    def affine_norm_const(
        self,
        r_ir: float,
        r_uv: float,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """
        Compute Z = ∫_{r_ir}^{r_uv} 1/ω(s) ds.

        Document eq (ads-affine-param):
        For planar slicing: ω(s) = e^{ds}, so
            Z = (e^{-d·r_ir} - e^{-d·r_uv}) / d

        For other slicings: uses cached trapezoidal integration.

        Args:
            r_ir: IR cutoff radius
            r_uv: UV cutoff radius
            device: Computation device (default: self.device)
            dtype: Computation dtype (default: self.dtype)

        Returns:
            Normalization constant Z

        Raises:
            ValueError: If r_uv <= r_ir
        """
        if r_uv <= r_ir:
            raise ValueError(f"Require r_uv > r_ir, got r_ir={r_ir}, r_uv={r_uv}")
        device = device if device is not None else self.device
        dtype = dtype if dtype is not None else self.dtype

        if self.slice_geometry == SliceGeometry.PLANAR:
            # Closed form for planar: ∫ e^{-ds} ds
            d = self.d
            Z = (math.exp(-d * r_ir) - math.exp(-d * r_uv)) / d
            return torch.tensor(Z, device=device, dtype=dtype)

        # For other geometries, use cached trapezoidal integration
        cache = self._get_affine_cache(r_ir, r_uv, device=device, dtype=dtype)
        return cache["Z"]

    def affine_parameter(self, r: Tensor, r_ir: float, r_uv: float) -> Tensor:
        """
        Return u(r) ∈ [0,1] for r ∈ [r_ir, r_uv].

        Document eq (ads-affine-param):
            u(r) = (1/Z) ∫_{r_ir}^r 1/ω(s) ds

        For planar slicing: closed form.
        For other geometries: cached interpolation.

        Args:
            r: Radial coordinate(s)
            r_ir: IR cutoff radius
            r_uv: UV cutoff radius

        Returns:
            Affine parameter u(r) ∈ [0, 1]
        """
        r = _as_tensor(r, device=self.device, dtype=self.dtype)

        if self.slice_geometry == SliceGeometry.PLANAR:
            # Numerically stable form: multiply top/bottom by e^{d*r_ir}
            # Avoids subtracting tiny exponentials for large r bounds
            d = self.d
            num = 1.0 - torch.exp(d * (r_ir - r))
            den = 1.0 - math.exp(d * (r_ir - r_uv))
            return num / den

        cache = self._get_affine_cache(
            float(r_ir), float(r_uv), device=r.device, dtype=r.dtype
        )
        return self._interp1d(cache["r_grid"], cache["u_grid"], r)

    def du_dr(self, r: Tensor, r_ir: float, r_uv: float) -> Tensor:
        """
        Compute du/dr = 1/(Z ω(r)).

        Document eq (ads-affine-param) derivative.

        Args:
            r: Radial coordinate(s)
            r_ir: IR cutoff radius
            r_uv: UV cutoff radius

        Returns:
            du/dr
        """
        r = _as_tensor(r, device=self.device, dtype=self.dtype)
        Z = self.affine_norm_const(r_ir, r_uv, device=r.device, dtype=r.dtype)
        return 1.0 / (Z * self.slice_weight(r))

    def dr_du(self, r: Tensor, r_ir: float, r_uv: float) -> Tensor:
        """
        Compute dr/du = Z ω(r).

        Inverse of du/dr, used in endpoint tangent conversion.

        Args:
            r: Radial coordinate(s)
            r_ir: IR cutoff radius
            r_uv: UV cutoff radius

        Returns:
            dr/du
        """
        r = _as_tensor(r, device=self.device, dtype=self.dtype)
        Z = self.affine_norm_const(r_ir, r_uv, device=r.device, dtype=r.dtype)
        return Z * self.slice_weight(r)

    # =========================================================================
    # Internal Cache Helpers for Affine Parameter
    # =========================================================================

    def _get_affine_cache(
        self,
        r_ir: float,
        r_uv: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, Tensor]:
        """
        Build or retrieve cached trapezoidal integration for affine parameter.

        Uses caching to avoid recomputing the integral for the same (r_ir, r_uv).

        Args:
            r_ir: IR cutoff radius
            r_uv: UV cutoff radius
            device: Computation device
            dtype: Computation dtype

        Returns:
            Cache dictionary with keys "r_grid", "u_grid", "Z"
        """
        key = (float(r_ir), float(r_uv), device, dtype)
        if key in self._affine_cache:
            return self._affine_cache[key]

        n = int(self.cfg.affine_cache_size)
        margin = float(self.cfg.affine_cache_margin)
        r0 = float(r_ir) - margin
        r1 = float(r_uv) + margin
        r_grid = torch.linspace(r0, r1, n, device=device, dtype=dtype)

        # Trapezoidal integration of 1/ω
        omega = self.slice_weight(r_grid)
        inv_omega = 1.0 / omega
        dr = (r1 - r0) / (n - 1)
        cum = torch.cumsum((inv_omega[:-1] + inv_omega[1:]) * 0.5 * dr, dim=0)
        cum = torch.cat([torch.zeros(1, device=device, dtype=dtype), cum], dim=0)

        # Normalize to [0,1] on [r_ir, r_uv]
        idx_ir = int(torch.argmin(torch.abs(r_grid - r_ir)).item())
        idx_uv = int(torch.argmin(torch.abs(r_grid - r_uv)).item())
        Z = (cum[idx_uv] - cum[idx_ir]).clamp_min(torch.finfo(dtype).tiny)
        u_grid = (cum - cum[idx_ir]) / Z

        cached = {"r_grid": r_grid, "u_grid": u_grid, "Z": Z}
        self._affine_cache[key] = cached
        return cached

    @staticmethod
    def _interp1d(x_grid: Tensor, y_grid: Tensor, x: Tensor) -> Tensor:
        """
        Vectorized 1D linear interpolation.

        Args:
            x_grid: Monotonically increasing x values
            y_grid: Corresponding y values
            x: Query points

        Returns:
            Interpolated y values
        """
        x = _as_tensor(x, device=x_grid.device, dtype=x_grid.dtype)
        x_flat = x.reshape(-1)
        x_flat = torch.clamp(x_flat, x_grid[0], x_grid[-1])

        idx = torch.searchsorted(x_grid, x_flat, right=True) - 1
        idx = torch.clamp(idx, 0, x_grid.numel() - 2)

        x_lo, x_hi = x_grid[idx], x_grid[idx + 1]
        y_lo, y_hi = y_grid[idx], y_grid[idx + 1]

        t = (x_flat - x_lo) / (x_hi - x_lo + torch.finfo(x_grid.dtype).tiny)
        y = y_lo + t * (y_hi - y_lo)
        return y.view_as(x)

    # =========================================================================
    # Quadrature Helpers (Document Section 2.1)
    # =========================================================================

    @staticmethod
    def unit_quadrature_weights_planar(
        grid_shape: Tuple[int, ...],
        box_lengths: Union[float, Tuple[float, ...]],
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """
        Unit-slice (dvol_ĝ) quadrature weights for a periodic planar box.

        Returns constant weight (Δx)^d for each grid point.

        Args:
            grid_shape: Spatial grid dimensions
            box_lengths: Domain size (scalar or per-dimension)
            device: Output device
            dtype: Output dtype

        Returns:
            Quadrature weights with shape grid_shape

        Raises:
            ValueError: If box_lengths length doesn't match grid_shape
        """
        device = torch.device("cpu") if device is None else device
        dtype = torch.float32 if dtype is None else dtype

        if isinstance(box_lengths, (int, float)):
            box_lengths = tuple(float(box_lengths) for _ in range(len(grid_shape)))
        if len(box_lengths) != len(grid_shape):
            raise ValueError("box_lengths length must match grid_shape length")

        dxs = [L / n for L, n in zip(box_lengths, grid_shape)]
        w0 = float(torch.tensor(dxs).prod().item())
        return torch.full(grid_shape, w0, device=device, dtype=dtype)

    @staticmethod
    def unit_quadrature_weights_s2_latlon(
        n_theta: int,
        n_phi: int,
        *,
        midpoint: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """
        Unit-slice weights for S² in (θ, φ) coordinates.

        weights = sin(θ) dθ dφ on the unit sphere.

        Args:
            n_theta: Number of latitude points
            n_phi: Number of longitude points
            midpoint: Use midpoint rule (vs endpoints)
            device: Output device
            dtype: Output dtype

        Returns:
            Quadrature weights with shape (n_theta, n_phi)

        Raises:
            ValueError: If n_theta or n_phi <= 0, or endpoint with n_theta=1
        """
        device = torch.device("cpu") if device is None else device
        dtype = torch.float32 if dtype is None else dtype

        if n_theta <= 0 or n_phi <= 0:
            raise ValueError("n_theta and n_phi must be positive")

        dphi = 2.0 * math.pi / n_phi
        if midpoint:
            dtheta = math.pi / n_theta
            theta = (torch.arange(n_theta, device=device, dtype=dtype) + 0.5) * dtheta
        else:
            if n_theta == 1:
                raise ValueError("Endpoint grid requires n_theta >= 2")
            dtheta = math.pi / (n_theta - 1)
            theta = torch.linspace(0.0, math.pi, n_theta, device=device, dtype=dtype)

        w_theta = torch.sin(theta).clamp_min(torch.finfo(dtype).tiny) * dtheta * dphi
        return w_theta[:, None].expand(n_theta, n_phi).contiguous()

    def slice_quadrature_weights_from_unit(
        self, w_unit: Tensor, r: Tensor
    ) -> Tensor:
        """
        Given unit-slice weights for dvol_ĝ, return weights for dvol_{g_r}.

        Document eq (slice-volume): dvol_{g_r} = ω(r) dvol_ĝ

        Args:
            w_unit: Unit-slice quadrature weights
            r: Radial coordinate

        Returns:
            Scaled quadrature weights
        """
        r = _as_tensor(r, device=w_unit.device, dtype=w_unit.dtype)
        omega = self.slice_weight(r)
        while omega.ndim < w_unit.ndim:
            omega = omega.unsqueeze(-1)
        return w_unit * omega

    # =========================================================================
    # Spectral Eigenvalue Helpers
    # =========================================================================

    @staticmethod
    def fft_laplacian_eigenvalues(
        shape: Tuple[int, ...],
        lengths: Union[float, Tuple[float, ...]],
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """
        Eigenvalues λ(k) = ||k||² for -Δ on a periodic box with FFT modes.

        Document eq (lap-eigs): -Δ_ĝ ψ_j = λ_j ψ_j, λ_j ≥ 0.

        Args:
            shape: Grid dimensions
            lengths: Domain size (scalar or per-dimension)
            device: Output device
            dtype: Output dtype

        Returns:
            Eigenvalue array with shape `shape`

        Raises:
            ValueError: If lengths doesn't match shape
        """
        device = torch.device("cpu") if device is None else device
        dtype = torch.float32 if dtype is None else dtype

        if isinstance(lengths, (int, float)):
            lengths = tuple(float(lengths) for _ in range(len(shape)))
        if len(lengths) != len(shape):
            raise ValueError("lengths must have same length as shape")

        grids = []
        for n, L in zip(shape, lengths):
            dx = L / n
            k = 2 * math.pi * torch.fft.fftfreq(n, d=dx, device=device, dtype=dtype)
            grids.append(k)

        mesh = torch.meshgrid(*grids, indexing="ij")
        lam = sum(m ** 2 for m in mesh)
        return lam


# =============================================================================
# Concrete Geometry Implementations (Document Definition 2.1)
# =============================================================================


@register_geometry("planar")
class PlanarAdS(AdSGeometry):
    """
    Planar slicing: f(r) = e^r.

    Document Section 2.1:
        ds² = dr² + e^{2r} dx²
        f(r) = e^r
        f'/f = 1
        ω(r) = e^{dr}

    This is the most commonly used slicing, where the boundary
    is conformally equivalent to flat R^d.
    """

    def f(self, r: Tensor) -> Tensor:
        """Warp factor f(r) = e^r."""
        r = _as_tensor(r, device=self.device, dtype=self.dtype)
        return torch.exp(r)

    def log_f_prime(self, r: Tensor) -> Tensor:
        """Logarithmic derivative f'/f = 1."""
        r = _as_tensor(r, device=self.device, dtype=self.dtype)
        return torch.ones_like(r)

    # Multi-warp stubs (not applicable for single-warp geometry)
    def a(self, r: Tensor) -> Tensor:
        raise NotImplementedError("PlanarAdS is single-warp; a(r) undefined")

    def b(self, r: Tensor) -> Tensor:
        raise NotImplementedError("PlanarAdS is single-warp; b(r) undefined")

    def log_a_prime(self, r: Tensor) -> Tensor:
        raise NotImplementedError("PlanarAdS is single-warp; log_a_prime undefined")

    def log_b_prime(self, r: Tensor) -> Tensor:
        raise NotImplementedError("PlanarAdS is single-warp; log_b_prime undefined")


@register_geometry("flat")
class FlatGeometry(AdSGeometry):
    """
    ABLATION: Flat Euclidean space with f(r) = 1.

    This is NOT AdS! It's flat (d+1)-dimensional Euclidean space:
        ds² = dr² + dx²  (no warping)

    Used for ablation studies to test whether AdS curvature matters.

    Properties:
        f(r) = 1           (constant, no warping)
        f'(r) = 0          (no radial dependence)
        f'/f = 0           (no friction)
        ω(r) = f(r)^d = 1  (constant slice volume)
        κ(r) = d·(f'/f) = 0 (no radial friction in KG equation)

    The Klein-Gordon equation simplifies to:
        ∂²Φ/∂r² + Δ_Σ Φ - m²Φ = 0

    This is just a simple wave equation without the AdS friction term!
    The affine parameter becomes linear: u(r) = r - r_IR.
    """

    def f(self, r: Tensor) -> Tensor:
        """Warp factor f(r) = 1 (constant)."""
        r = _as_tensor(r, device=self.device, dtype=self.dtype)
        return torch.ones_like(r)

    def log_f_prime(self, r: Tensor) -> Tensor:
        """Logarithmic derivative f'/f = 0."""
        r = _as_tensor(r, device=self.device, dtype=self.dtype)
        return torch.zeros_like(r)

    # Multi-warp stubs
    def a(self, r: Tensor) -> Tensor:
        raise NotImplementedError("FlatGeometry is single-warp; a(r) undefined")

    def b(self, r: Tensor) -> Tensor:
        raise NotImplementedError("FlatGeometry is single-warp; b(r) undefined")

    def log_a_prime(self, r: Tensor) -> Tensor:
        raise NotImplementedError("FlatGeometry is single-warp; log_a_prime undefined")

    def log_b_prime(self, r: Tensor) -> Tensor:
        raise NotImplementedError("FlatGeometry is single-warp; log_b_prime undefined")


@register_geometry("planar_hsv")
class HyperscalingViolatingAdS(AdSGeometry):
    """
    Hyperscaling-violating (HSV) geometry: one-parameter family interpolating
    between flat space and AdS (paper Section 5).

    CONVENTION NOTE: The code uses p ∈ [0, 1) with p=0 flat and p→1 AdS.
    This is INVERTED from the paper's convention (paper eq. 43 uses p=0 for AdS).

    Paper reference (with inverted p):
        Original metric: ds² = z^{-2(1-p)}(dz² + dx²)  [paper eq. 43]
        
        After coordinate transform r = z^p/p:
        ds² = dr² + (pr)^{-2γ} dx²,  γ = 1/p - 1  [paper eq. 44]

    Code parameterization (inverted convention):
        Warp factor: f_p(r) = [(1-p)r]^{-p/(1-p)}    for p ∈ [0, 1)

    Special cases:
        p = 0:  f(r) = 1           (flat space)
        p → 1:  f(r) → e^r         (planar AdS limit)

    Key formulas:
        f'/f = -p / [(1-p)r]
        ω(r) = f(r)^d = [(1-p)r]^{-α}  where α = pd/(1-p)

    Affine coordinate for Hermite path (paper eq. 48 adapted):
        u_p(r) = (r^γ - r_IR^γ) / (r_UV^γ - r_IR^γ)
        γ = 1 + α = 1 + pd/(1-p) = [1 + (d-1)p] / (1-p)

    HSV propagator (paper eq. 47):
        K̂(u, k) = (|k|u)^β K_β(|k|u) / (2^{β-1} Γ(β))
        β = (1 + (d-1)p) / 2  (Bessel order)

    Numerical handling:
        - For p < threshold (default 0.95): use HSV formulas
        - For p >= threshold: switch to exact AdS formulas
        - Requires r > 0 (singularity at r=0 for p > 0)

    References:
        - Paper Section 5: "Interpolating between flat space and AdS"
        - Huijse et al., arXiv:1112.1548 (Hyperscaling violation in gauge/gravity)
    """

    def __init__(self, cfg: GeometryConfig) -> None:
        """
        Initialize HSV geometry.

        Args:
            cfg: Geometry configuration with hsv_p parameter

        Raises:
            ValueError: If hsv_p is not in [0, 1)
        """
        super().__init__(cfg)
        self._p = cfg.hsv_p
        self._threshold = cfg.hsv_ads_threshold

        if not 0.0 <= self._p < 1.0:
            raise ValueError(f"HSV parameter p must be in [0, 1), got {self._p}")

        # Precompute useful quantities
        self._use_ads = self._p >= self._threshold
        if not self._use_ads:
            self._one_minus_p = 1.0 - self._p
            # Exponent: -p/(1-p)
            self._exponent = -self._p / self._one_minus_p
            # Alpha for slice weight: α = pd/(1-p)
            self._alpha = self._p * self.d / self._one_minus_p
            # Gamma for affine coordinate: γ = 1 + α
            self._gamma = 1.0 + self._alpha

    @property
    def p(self) -> float:
        """HSV parameter p ∈ [0, 1)."""
        return self._p

    @property
    def gamma(self) -> float:
        """Affine coordinate exponent γ = 1 + pd/(1-p)."""
        if self._use_ads:
            return float('inf')  # AdS limit
        return self._gamma

    @property
    def alpha(self) -> float:
        """Slice weight exponent α = pd/(1-p)."""
        if self._use_ads:
            return float('inf')  # AdS limit
        return self._alpha

    @property
    def is_hsv(self) -> bool:
        """This is a hyperscaling-violating geometry."""
        return True

    def log_f(self, r: Tensor) -> Tensor:
        """
        Compute σ = log f(r) for stable HSV computations.
        
        For HSV: σ = -p/(1-p) * log((1-p)*r)
        For AdS limit: σ = r
        """
        r = _as_tensor(r, device=self.device, dtype=self.dtype)
        
        if self._use_ads:
            return r
        
        if self._p == 0.0:
            return torch.zeros_like(r)
        
        r_safe = r.clamp(min=self.cfg.eps)
        return self._exponent * torch.log(self._one_minus_p * r_safe)

    def r_from_sigma(self, sigma: float) -> float:
        """
        Compute r given σ = log f(r).
        
        For HSV: r = exp(-(1-p)/p * σ) / (1-p)
        For AdS limit: r = σ
        
        Args:
            sigma: Target log-warp-factor value
            
        Returns:
            Radial coordinate r such that log f(r) = σ
        """
        if self._use_ads:
            return sigma
        
        if self._p < 1e-10:
            # p ≈ 0: f ≈ 1, σ ≈ 0 always, return arbitrary positive r
            return 1.0
        
        # σ = -p/(1-p) * log((1-p)*r)
        # => log((1-p)*r) = -σ*(1-p)/p
        # => (1-p)*r = exp(-σ*(1-p)/p)
        # => r = exp(-σ*(1-p)/p) / (1-p)
        return math.exp(-sigma * self._one_minus_p / self._p) / self._one_minus_p

    def u_to_r(self, u: float) -> float:
        """
        Convert HSV coordinate u to radial coordinate r.
        
        u = [(1-p)r]^{1/(1-p)}  =>  r = u^{1-p} / (1-p)
        
        Args:
            u: HSV coordinate (appears in propagator K̂(u,k))
            
        Returns:
            Radial coordinate r
        """
        if self._use_ads:
            # AdS limit: u = z = e^{-r}, so r = -log(u)
            return -math.log(max(u, 1e-10))
        
        if self._p < 1e-10:
            # Flat space: u = r
            return u
        
        # HSV: r = u^{1-p} / (1-p)
        return (u ** self._one_minus_p) / self._one_minus_p

    def get_r_bounds_from_u(
        self, u_uv: float, u_ir: float
    ) -> Tuple[float, float]:
        """
        Get r_ir, r_uv from HSV u-coordinate bounds.
        
        The u coordinate is natural for HSV:
        - Appears directly in propagator: K̂(u,k) = (|k|u)^β K_β(|k|u)
        - p=0: u = r (flat), p→1: u = z = e^{-r} (AdS)
        - Smaller u is UV (boundary), larger u is IR (bulk)
        
        HSV coordinate convention:
        - f(r) = [(1-p)r]^{-p/(1-p)} diverges as r→0
        - Small r = large f = UV (boundary)
        - Large r = small f = IR (bulk)
        - Therefore r_uv < r_ir for HSV (opposite of standard AdS)
        
        Args:
            u_uv: UV boundary in u-space (smaller u, near boundary)
            u_ir: IR boundary in u-space (larger u, deep bulk)
            
        Returns:
            Tuple of (r_ir, r_uv) where r_uv < r_ir for HSV
        """
        r_uv = self.u_to_r(u_uv)
        r_ir = self.u_to_r(u_ir)
        # NO SWAP: For HSV, r_uv < r_ir is correct (small r is UV)
        return r_ir, r_uv

    def get_r_bounds_from_sigma(
        self, sigma_min: float, sigma_max: float
    ) -> Tuple[float, float]:
        """
        Get r_ir, r_uv from sigma bounds.
        
        For HSV, f(r) decreases as r increases, so:
        - σ_min (smaller σ) corresponds to r_uv (larger r, UV)
        - σ_max (larger σ) corresponds to r_ir (smaller r, IR)
        
        This ensures ω = f^d = exp(d*σ) stays bounded regardless of p.
        
        Args:
            sigma_min: Minimum σ = log f (at UV boundary)
            sigma_max: Maximum σ = log f (at IR boundary)
            
        Returns:
            Tuple of (r_ir, r_uv) computed from sigma bounds
        """
        r_uv = self.r_from_sigma(sigma_min)  # UV: smaller σ → larger r
        r_ir = self.r_from_sigma(sigma_max)  # IR: larger σ → smaller r
        return r_ir, r_uv

    def f(self, r: Tensor) -> Tensor:
        """
        Warp factor f_p(r) = [(1-p)r]^(-p/(1-p)).

        For p >= threshold, switches to AdS: f(r) = e^r.

        Args:
            r: Radial coordinate (must be > 0 for p > 0)

        Returns:
            Warp factor f(r)
        """
        r = _as_tensor(r, device=self.device, dtype=self.dtype)

        if self._use_ads:
            # AdS limit: f(r) = e^r
            return torch.exp(r)

        if self._p == 0.0:
            # Flat space: f(r) = 1
            return torch.ones_like(r)

        # HSV: f(r) = [(1-p)r]^(-p/(1-p))
        # Clamp r to avoid log of zero/negative
        r_safe = r.clamp(min=self.cfg.eps)
        base = self._one_minus_p * r_safe
        return torch.pow(base, self._exponent)

    def log_f_prime(self, r: Tensor) -> Tensor:
        """
        Logarithmic derivative f'/f = -p / [(1-p)r].

        For p >= threshold, switches to AdS: f'/f = 1.

        Args:
            r: Radial coordinate

        Returns:
            f'/f at r
        """
        r = _as_tensor(r, device=self.device, dtype=self.dtype)

        if self._use_ads:
            # AdS limit: f'/f = 1
            return torch.ones_like(r)

        if self._p == 0.0:
            # Flat space: f'/f = 0
            return torch.zeros_like(r)

        # HSV: f'/f = -p / [(1-p)r]
        r_safe = r.clamp(min=self.cfg.eps)
        return -self._p / (self._one_minus_p * r_safe)

    def slice_weight(self, r: Tensor) -> Tensor:
        """
        Slice volume weight ω(r) = f(r)^d = [(1-p)r]^(-α).

        For p >= threshold, switches to AdS: ω(r) = e^(dr).

        Args:
            r: Radial coordinate

        Returns:
            Volume weight ω(r)
        """
        r = _as_tensor(r, device=self.device, dtype=self.dtype)

        if self._use_ads:
            # AdS limit: ω(r) = e^(dr)
            return torch.exp(self.d * r)

        if self._p == 0.0:
            # Flat space: ω(r) = 1
            return torch.ones_like(r)

        # HSV: ω(r) = [(1-p)r]^(-α) where α = pd/(1-p)
        r_safe = r.clamp(min=self.cfg.eps)
        base = self._one_minus_p * r_safe
        return torch.pow(base, -self._alpha)

    def affine_parameter(self, r: Tensor, r_ir: float, r_uv: float) -> Tensor:
        """
        Explicit affine parameter u_p(r) for HSV geometry.

        u_p(r) = (r^γ - r_IR^γ) / (r_UV^γ - r_IR^γ)

        where γ = 1 + pd/(1-p).

        For p >= threshold, uses AdS formula.
        For p = 0, reduces to linear: u = (r - r_IR) / (r_UV - r_IR).

        Args:
            r: Radial coordinate(s)
            r_ir: IR cutoff radius
            r_uv: UV cutoff radius

        Returns:
            Affine parameter u(r) ∈ [0, 1]
        """
        r = _as_tensor(r, device=self.device, dtype=self.dtype)

        if self._use_ads:
            # AdS limit: use exponential formula (numerically stable form)
            d = self.d
            num = 1.0 - torch.exp(d * (r_ir - r))
            den = 1.0 - math.exp(d * (r_ir - r_uv))
            return num / den

        if self._p == 0.0:
            # Flat space: linear
            return (r - r_ir) / (r_uv - r_ir)

        # HSV: u_p(r) = (r^γ - r_IR^γ) / (r_UV^γ - r_IR^γ)
        gamma = self._gamma
        r_ir_g = r_ir ** gamma
        r_uv_g = r_uv ** gamma
        r_g = torch.pow(r.clamp(min=self.cfg.eps), gamma)
        return (r_g - r_ir_g) / (r_uv_g - r_ir_g)

    def dr_du(self, r: Tensor, r_ir: float, r_uv: float) -> Tensor:
        """
        Jacobian dr/du = Z * ω(r) for endpoint tangent computation.

        For HSV:
            dr/du = (r_UV^γ - r_IR^γ) / γ * [(1-p)r]^(-α) / (1-p)^α
                  = (r_UV^γ - r_IR^γ) / (γ * r^α)

        Args:
            r: Radial coordinate(s)
            r_ir: IR cutoff radius
            r_uv: UV cutoff radius

        Returns:
            dr/du at r
        """
        r = _as_tensor(r, device=self.device, dtype=self.dtype)

        if self._use_ads:
            # AdS limit: dr/du = Z * e^(dr)
            # Z = (e^{-d*r_ir} - e^{-d*r_uv})/d = e^{-d*r_ir}(1 - e^{d(r_ir-r_uv)})/d
            # Numerically stable: Z = (1 - e^{d(r_ir-r_uv)}) / (d * e^{d*r_ir})
            # dr/du = Z * e^{d*r} = (1 - e^{d(r_ir-r_uv)}) * e^{d*(r-r_ir)} / d
            d = self.d
            Z_scaled = (1.0 - math.exp(d * (r_ir - r_uv))) / d
            return Z_scaled * torch.exp(d * (r - r_ir))

        if self._p == 0.0:
            # Flat space: dr/du = r_UV - r_IR (constant)
            return torch.full_like(r, r_uv - r_ir)

        # HSV: dr/du = (r_UV^γ - r_IR^γ) / (γ * r^α)
        gamma = self._gamma
        alpha = self._alpha
        r_ir_g = r_ir ** gamma
        r_uv_g = r_uv ** gamma
        r_safe = r.clamp(min=self.cfg.eps)
        return (r_uv_g - r_ir_g) / (gamma * torch.pow(r_safe, alpha))

    def normalization_Z(self, r_ir: float, r_uv: float) -> float:
        """
        Normalization constant Z = ∫_{r_IR}^{r_UV} dr/ω(r).

        For HSV:
            Z = (1-p)^α * (r_UV^γ - r_IR^γ) / γ

        Args:
            r_ir: IR cutoff radius
            r_uv: UV cutoff radius

        Returns:
            Normalization constant Z
        """
        if self._use_ads:
            # AdS limit: Z = (e^{-d*r_ir} - e^{-d*r_uv})/d
            # Numerically stable: Z = e^{-d*r_ir} * (1 - e^{d(r_ir-r_uv)}) / d
            # For numerical purposes, we can use the stable form directly
            d = self.d
            return math.exp(-d * r_ir) * (1.0 - math.exp(d * (r_ir - r_uv))) / d

        if self._p == 0.0:
            # Flat space
            return r_uv - r_ir

        # HSV: Z = (1-p)^α * (r_UV^γ - r_IR^γ) / γ
        gamma = self._gamma
        alpha = self._alpha
        r_ir_g = r_ir ** gamma
        r_uv_g = r_uv ** gamma
        return (self._one_minus_p ** alpha) * (r_uv_g - r_ir_g) / gamma

    def affine_norm_const(
        self,
        r_ir: float,
        r_uv: float,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """
        Compute Z = ∫_{r_IR}^{r_UV} 1/ω(s) ds using closed-form HSV formula.

        This overrides the base class method to use the exact analytical
        expression instead of numerical integration.

        Args:
            r_ir: IR cutoff radius
            r_uv: UV cutoff radius
            device: Computation device (default: self.device)
            dtype: Computation dtype (default: self.dtype)

        Returns:
            Normalization constant Z as a tensor
        """
        if r_uv <= r_ir:
            raise ValueError(f"Require r_uv > r_ir, got r_ir={r_ir}, r_uv={r_uv}")
        device = device if device is not None else self.device
        dtype = dtype if dtype is not None else self.dtype

        Z = self.normalization_Z(r_ir, r_uv)
        return torch.tensor(Z, device=device, dtype=dtype)

    # Multi-warp stubs (HSV is single-warp)
    def a(self, r: Tensor) -> Tensor:
        raise NotImplementedError("HyperscalingViolatingAdS is single-warp; a(r) undefined")

    def b(self, r: Tensor) -> Tensor:
        raise NotImplementedError("HyperscalingViolatingAdS is single-warp; b(r) undefined")

    def log_a_prime(self, r: Tensor) -> Tensor:
        raise NotImplementedError("HyperscalingViolatingAdS is single-warp; log_a_prime undefined")

    def log_b_prime(self, r: Tensor) -> Tensor:
        raise NotImplementedError("HyperscalingViolatingAdS is single-warp; log_b_prime undefined")

    def __repr__(self) -> str:
        return (
            f"HyperscalingViolatingAdS(d={self.d}, p={self._p:.4f}, "
            f"gamma={self._gamma if not self._use_ads else 'inf':.4f}, "
            f"use_ads={self._use_ads})"
        )


# =============================================================================
# Factory Function
# =============================================================================


def make_geometry(cfg: GeometryConfig) -> AdSGeometry:
    """
    Factory function to create the appropriate AdSGeometry subclass.

    This is the recommended way to create geometry instances, as it
    handles the mapping from SliceGeometry enum to concrete class.

    Args:
        cfg: Geometry configuration

    Returns:
        Appropriate AdSGeometry subclass instance

    Raises:
        ValueError: If slice_geometry is not recognized

    Example:
        >>> cfg = GeometryConfig(d=2, slice_geometry=SliceGeometry.PLANAR)
        >>> geometry = make_geometry(cfg)
        >>> print(type(geometry).__name__)
        PlanarAdS
    """
    geometry_map = {
        SliceGeometry.PLANAR: PlanarAdS,
        SliceGeometry.FLAT: FlatGeometry,
        SliceGeometry.HYPERSCALING_VIOLATING: HyperscalingViolatingAdS,
    }

    if cfg.slice_geometry not in geometry_map:
        raise ValueError(f"Unknown slice geometry: {cfg.slice_geometry}")

    return geometry_map[cfg.slice_geometry](cfg)
