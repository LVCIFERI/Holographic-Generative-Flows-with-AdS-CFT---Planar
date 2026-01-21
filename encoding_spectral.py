"""
encoding_spectral.py

Geometry-Aware Spectral Encoding for AdS/CFT Flow Matching
===========================================================

This module provides spectral holographic encoding that converts boundary
point data into bulk field representations with correct AdS/CFT physics.

Following paper Section 3.3 (Spectral representation of data), the key insight
is that the bulk-boundary propagator determines how boundary sources create
bulk fields. In Fourier space (paper eq. 22):

    κ̃_{|k|}(r) = (2/Γ(ν)) (|k|e^r/2)^ν K_ν(|k|e^{-r})

where ν = Δ - d/2 and K_ν is the modified Bessel function of the second kind.

Spectral Point Encoding (Paper Section 3.3)
-------------------------------------------
Data points x_* are encoded as delta-function sources:
    J(x|x_*) = N_ν δ(x - x_*)

The Fourier mode coefficients become:
    φ̃_k(r|x_*) = (N_ν / (2π)^{d/2}) κ̃_{|k|}(r) e^{ik·x_*}

The phase e^{ik·x_*} encodes the position via the Fourier shift theorem.

Supported Geometries
--------------------
- PLANAR: Fourier basis with Bessel propagator (exact AdS/CFT, paper eq. 22)
- FLAT: Flat Euclidean propagator e^{-|k|r} (ablation baseline)
- PLANAR_HSV: Hyperscaling-violating with unified Bessel form (paper eq. 47)

Encoding Approaches
-------------------
1. SpectralHolographicEncoder: Grid-based spectral encoding
   - Memory: O(K²) for K spectral modes
   - Phase encodes position (Fourier shift theorem)

2. HolographicPointEncoder: Grid-free analytic encoding
   - Memory: O(C) per sample (minimal)
   - Preserves exact coordinates
   - Laplacian effect computed analytically

3. HolographicPointFieldEncoder: Full propagator encoding
   - Memory: O(H×W) for grid
   - Exact bulk-boundary propagator on grid
   - Most physically faithful

Decoding Methods
----------------
- Phase gradient: Sub-grid precision via Fourier shift theorem
- Soft-argmax: Robust grid-based decoding
- Density sampling: CFT-faithful probabilistic decoding

Paper References
----------------
- Paper Section 2: Klein-Gordon theory in AdS/CFT
- Paper Section 2.1: Solving Klein-Gordon with propagators
- Paper Section 2.2: Spectral decomposition of Klein-Gordon
- Paper Section 2.4: Boundary stabilization of fields (eq. 22)
- Paper Section 3.3: Spectral representation of data
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ads_cft.registry import register_encoder
from ads_cft.geometry import SliceGeometry
from ads_cft.encoding_base import (
    BaseEncoder,
    BulkField,
    _bessel_k_nu,
    SCIPY_AVAILABLE,
)

Tensor = torch.Tensor


# =============================================================================
# Spectral Holographic Encoder
# =============================================================================


@register_encoder("spectral")
class SpectralHolographicEncoder(nn.Module):
    """
    Geometry-aware spectral holographic encoding for AdS/CFT bulk-boundary propagator.

    Supports slice geometries:
    - PLANAR: Fourier basis with Bessel-based propagator K̂_Δ(z, k) ∝ |k|^ν K_ν(|k|z)
    - FLAT: Flat Euclidean propagator (ablation baseline)
    - HYPERSCALING_VIOLATING: HSV propagator with p-dependent scaling

    For planar geometry, uses exact Bessel propagator.

    Attributes
    ----------
    d : int
        Boundary dimension
    n_channels : int
        Number of field channels (one per conformal dimension)
    n_modes : int
        Number of spectral modes per dimension
    r_uv : float
        UV boundary radial coordinate
    domain : Tuple[float, ...]
        Domain bounds (x_min, x_max, y_min, y_max)
    geometry : SliceGeometry
        Slice geometry type
    envelopes_phi : Tensor
        Shape (C, K, K) field propagator envelopes
    envelopes_pi : Tensor
        Shape (C, K, K) momentum propagator envelopes

    Memory: O(K²) for K modes
    Laplacian: Diagonal O(K²)

    Example
    -------
    >>> encoder = SpectralHolographicEncoder(
    ...     d=2, deltas=[1.5, 2.0], n_modes=16, r_uv=1.0,
    ...     geometry=SliceGeometry.PLANAR
    ... )
    >>> phi_hat, pi_hat = encoder(points)  # (B, 2C, K, K)
    >>> points_decoded = encoder.decode(phi_hat)
    """

    def __init__(
        self,
        d: int,
        deltas: Sequence[float],
        n_modes: int,
        r_uv: float,
        domain: Tuple[float, float, float, float] = (-4.0, 4.0, -4.0, 4.0),
        regularization: float = 1e-2,
        geometry: SliceGeometry = SliceGeometry.PLANAR,
        hsv_p: Optional[float] = None,
        hsv_propagator_mode: str = "bessel",
    ) -> None:
        """
        Initialize spectral holographic encoder.

        Args:
            d: Boundary dimension (2 for 2D points)
            deltas: Conformal dimensions Δ_i for each channel (must be > d/2)
            n_modes: Number of spectral modes per dimension
            r_uv: UV radius where lift occurs
            domain: (x_min, x_max, y_min, y_max) for planar
            regularization: Small ε for numerical stability
            geometry: Slice geometry (planar, flat, planar_hsv)
            hsv_p: HSV parameter p (only used for HSV geometries to select propagator)
            hsv_propagator_mode: "bessel" for closed-form conformal product, 
                                 "emd" for full EMD ODE integration
        """
        super().__init__()
        self.d = d
        self.n_channels = len(deltas)
        self.n_modes = n_modes
        self.r_uv = r_uv
        self.domain = domain
        self.regularization = regularization
        self.geometry = geometry
        self.hsv_p = hsv_p
        self.hsv_propagator_mode = hsv_propagator_mode

        deltas_t = torch.tensor(list(deltas), dtype=torch.float32)
        self.register_buffer("deltas", deltas_t)

        # Compute Bessel orders: ν_i = Δ_i - d/2
        nus = deltas_t - d / 2.0
        self.register_buffer("nus", nus)

        # Domain size (for planar)
        x_min, x_max, y_min, y_max = domain
        Lx = x_max - x_min
        Ly = y_max - y_min
        self.register_buffer("Lx", torch.tensor(Lx, dtype=torch.float32))
        self.register_buffer("Ly", torch.tensor(Ly, dtype=torch.float32))
        self.register_buffer("x_min", torch.tensor(x_min, dtype=torch.float32))
        self.register_buffer("y_min", torch.tensor(y_min, dtype=torch.float32))

        # =====================================================================
        # GEOMETRY-SPECIFIC SPECTRAL SETUP
        # =====================================================================

        if geometry == SliceGeometry.PLANAR:
            self._init_planar(n_modes, Lx, Ly, r_uv, regularization, deltas_t, nus)
        elif geometry == SliceGeometry.FLAT:
            # ABLATION: Use flat-space propagator (simple exponential decay)
            # For massive scalar in flat space: Φ_k(r) ∝ exp(-√(k²+m²) r)
            # This tests whether AdS curvature (Bessel propagator) matters
            self._init_flat(n_modes, Lx, Ly, r_uv, regularization, deltas_t)
        # HSV geometry: use p-dependent Bessel propagator
        # Unifies flat (p=0) and AdS (p→1) limits in a single formula
        elif geometry == SliceGeometry.HYPERSCALING_VIOLATING:
            self._init_hsv_planar(n_modes, Lx, Ly, r_uv, regularization, deltas_t, hsv_p)
        else:
            raise ValueError(f"Unknown geometry: {geometry}")

    def _init_planar(
        self,
        n_modes: int,
        Lx: float,
        Ly: float,
        r_uv: float,
        regularization: float,
        deltas_t: Tensor,
        nus: Tensor,
    ) -> None:
        """
        Initialize planar (Fourier) spectral basis with Bessel propagator.

        For planar AdS, the UV-stabilized bulk-boundary propagator in Fourier space is:
            tilde{K}_Δ(r, k) ∝ (|k|e^{-r})^ν K_ν(|k|e^{-r}) = ξ^ν K_ν(ξ)

        where ν = Δ - d/2, ξ = |k|z, and z = e^{-r}.

        The normalization N_Δ = 2^{1-ν}/Γ(ν) is r-independent.

        Document Reference:
            Section 2.4, equation (29): UV-stabilized propagator mode coefficients
        """
        # Wave numbers using FFT ordering
        kx_freq = torch.fft.fftfreq(n_modes) * n_modes
        ky_freq = torch.fft.fftfreq(n_modes) * n_modes

        kx = 2.0 * math.pi * kx_freq / Lx
        ky = 2.0 * math.pi * ky_freq / Ly

        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        k_sq = KX ** 2 + KY ** 2
        k_mag = torch.sqrt(k_sq + regularization ** 2)

        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.register_buffer("KX", KX)
        self.register_buffer("KY", KY)
        self.register_buffer("k_sq", k_sq)
        self.register_buffer("k_mag", k_mag)

        # Poincaré coordinate at UV
        z_uv = math.exp(-r_uv)
        self.register_buffer("z_uv", torch.tensor(z_uv, dtype=torch.float32))

        # Precompute Bessel-based envelopes for each channel
        envelopes_phi = []
        envelopes_pi = []

        for c in range(self.n_channels):
            delta = float(deltas_t[c])
            nu = float(nus[c])

            # xi = |k| * z = |k| * e^{-r} is the argument for Bessel functions
            xi = k_mag * z_uv
            K_nu = _bessel_k_nu(nu, xi, eps=regularization)

            # UV-stabilized propagator: K̃ = e^{Δr} K
            # Starting from κ_{|k|}(r) = (2/Γ(ν)) e^{-dr/2} (|k|/2)^ν K_ν(|k|e^{-r})
            # Multiplying by e^{Δr} gives:
            #   κ̃_{|k|}(r) = (2/Γ(ν)) e^{(Δ-d/2)r} (|k|/2)^ν K_ν(ξ)
            #              = (2/Γ(ν)) e^{νr} (|k|/2)^ν K_ν(ξ)
            # where ν = Δ - d/2 and ξ = |k|e^{-r}
            prefactor = math.exp(nu * r_uv)
            k_power = torch.pow(k_mag, nu)
            envelope_phi_raw = prefactor * k_power * K_nu

            norm_val = envelope_phi_raw[0, 0].item()
            if abs(norm_val) > 1e-10:
                envelope_phi = envelope_phi_raw / norm_val
            else:
                envelope_phi = envelope_phi_raw

            # Momentum envelope: Π̃ = e^{Δr}(Π + ΔΦ) = ∂_r Φ̃
            # ∂_r[e^{νr} |k|^ν K_ν(ξ)] = e^{νr} |k|^ν [ν K_ν(ξ) + ξ K_{ν-1}(ξ) · (+1)]
            #                          = e^{νr} |k|^ν [ν K_ν(ξ) - ξ K'_ν(ξ)]
            # Using K'_ν(ξ) = -K_{ν-1}(ξ) - (ν/ξ)K_ν(ξ), we get:
            #   ∂_r Φ̃ = e^{νr} |k|^ν [2ν K_ν(ξ) + ξ K_{ν-1}(ξ)]
            K_nu_minus_1 = _bessel_k_nu(abs(nu - 1.0), xi, eps=regularization)
            envelope_pi_raw = prefactor * k_power * (xi * K_nu_minus_1 + 2.0 * nu * K_nu)

            if abs(norm_val) > 1e-10:
                envelope_pi = envelope_pi_raw / norm_val
            else:
                envelope_pi = envelope_pi_raw

            envelopes_phi.append(envelope_phi)
            envelopes_pi.append(envelope_pi)

        self.register_buffer("envelopes_phi", torch.stack(envelopes_phi, dim=0))
        self.register_buffer("envelopes_pi", torch.stack(envelopes_pi, dim=0))

    def _init_flat(
        self,
        n_modes: int,
        Lx: float,
        Ly: float,
        r_uv: float,
        regularization: float,
        deltas_t: Tensor,
    ) -> None:
        """
        Initialize flat Euclidean space spectral basis (ABLATION).

        For massive scalar in flat (d+1) Euclidean space:
            (∂²/∂r² + Δ_Σ - m²)Φ = 0

        In Fourier space:
            (∂²/∂r² - k² - m²)Φ_k = 0

        Solution (decaying into bulk):
            Φ_k(r) = A_k exp(-μ r)  where μ = √(k² + m²)

        This is the ABLATION baseline: no AdS curvature, just flat space wave equation.
        The propagator is simple exponential, not Bessel function.

        With mass m² = Δ(Δ-d), we have μ = √(k² + Δ(Δ-d)).
        """
        # Wave numbers (same as planar)
        kx_freq = torch.fft.fftfreq(n_modes) * n_modes
        ky_freq = torch.fft.fftfreq(n_modes) * n_modes

        kx = 2.0 * math.pi * kx_freq / Lx
        ky = 2.0 * math.pi * ky_freq / Ly

        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        k_sq = KX ** 2 + KY ** 2
        k_mag = torch.sqrt(k_sq + regularization ** 2)

        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.register_buffer("KX", KX)
        self.register_buffer("KY", KY)
        self.register_buffer("k_sq", k_sq)
        self.register_buffer("k_mag", k_mag)

        # For flat space, z_uv doesn't have special meaning, but we keep it for compatibility
        z_uv = math.exp(-r_uv)
        self.register_buffer("z_uv", torch.tensor(z_uv, dtype=torch.float32))

        # Compute flat-space exponential envelopes for each channel
        envelopes_phi = []
        envelopes_pi = []

        d = 2  # boundary dimension

        for c in range(self.n_channels):
            delta = float(deltas_t[c])

            # Mass squared from conformal dimension: m² = Δ(Δ-d)
            m_sq = delta * (delta - d)

            # Effective "radial momentum": μ = √(k² + m²)
            # Add regularization for numerical stability at k=0
            mu = torch.sqrt(k_sq + max(m_sq, 0) + regularization ** 2)

            # Flat space propagator: exp(-μ r_UV)
            # This is the envelope for field amplitude at the boundary
            envelope_phi = torch.exp(-mu * r_uv)

            # Normalize so zero-mode has unit amplitude
            norm_val = envelope_phi[0, 0].item()
            if abs(norm_val) > 1e-10:
                envelope_phi = envelope_phi / norm_val

            # Momentum envelope: ∂_r Φ = -μ Φ
            envelope_pi_raw = -mu * torch.exp(-mu * r_uv)
            if abs(norm_val) > 1e-10:
                envelope_pi = envelope_pi_raw / norm_val
            else:
                envelope_pi = envelope_pi_raw

            envelopes_phi.append(envelope_phi)
            envelopes_pi.append(envelope_pi)

        self.register_buffer("envelopes_phi", torch.stack(envelopes_phi, dim=0))
        self.register_buffer("envelopes_pi", torch.stack(envelopes_pi, dim=0))

        print(
            f"[SPECTRAL-FLAT] ABLATION: Using flat Euclidean propagator (no AdS curvature)"
        )
        print(f"[SPECTRAL-FLAT] Propagator: exp(-√(k²+m²)·r), not Bessel functions")

    def _init_hsv_planar(
        self,
        n_modes: int,
        Lx: float,
        Ly: float,
        r_uv: float,
        regularization: float,
        deltas_t: Tensor,
        hsv_p: Optional[float],
    ) -> None:
        """
        Initialize HSV planar spectral basis with p-dependent Bessel propagator.

        The bulk-boundary propagator for hyperscaling-violating geometry is:

            K̂(u, k) = (|k|u)^β K_β(|k|u) / (2^{β-1} Γ(β))

        where:
            β = (1 + (d-1)p) / 2      (Bessel order)
            u = [(1-p)r]^{1/(1-p)}    (HSV radial coordinate)

        This formula unifies flat space and AdS limits:
            p = 0:  β = 1/2, u = r     →  K̂ = e^{-|k|r}           (flat)
            p → 1:  β → d/2, u → z     →  K̂ ∝ (|k|z)^{d/2} K_{d/2}  (AdS)

        For p >= 0.95 (near AdS), the HSV coordinate u becomes numerically
        unstable, so we fall back to the standard AdS Bessel propagator.

        The momentum propagator uses the Bessel identity:
            d/dx [x^β K_β(x)] = -x^β K_{β-1}(x)

        So:
            dK̂/dr = -|k| · (|k|u)^β K_{β-1}(|k|u) · u^p / normalization

        References:
            - Hyperscaling-violating holography (Dong et al., arXiv:1201.1905)
            - Probe scalars in HV backgrounds
        """
        from scipy.special import kv as scipy_kv
        from scipy.special import gamma as scipy_gamma

        d = self.d
        p = hsv_p if hsv_p is not None else 0.0
        
        # Threshold for switching to AdS formulas (numerical stability)
        HSV_ADS_THRESHOLD = 0.95
        
        # For p >= threshold, the HSV coordinate u = [(1-p)r]^{1/(1-p)} becomes
        # numerically unstable (approaches 0 extremely fast). Fall back to AdS.
        if p >= HSV_ADS_THRESHOLD:
            print(f"[SPECTRAL-HSV] p={p:.4f} >= {HSV_ADS_THRESHOLD}, using AdS Bessel propagator")
            # Compute nus for AdS Bessel (ν = Δ - d/2)
            nus = deltas_t - d / 2.0
            self._init_planar(n_modes, Lx, Ly, r_uv, regularization, deltas_t, nus)
            self.hsv_p = p
            self.hsv_beta = d / 2.0  # AdS limit
            return

        # Bessel order: β = (1 + (d-1)p) / 2
        beta = (1.0 + (d - 1) * p) / 2.0
        
        # KG gamma for UV stabilization: γ = p/(1-p)
        # This appears in the field redefinition Φ̃ = [(1-p)r]^{-dγ} Φ
        if p < 1e-10:
            kg_gamma = 0.0
        else:
            one_minus_p = 1.0 - p
            kg_gamma = p / one_minus_p

        # Compute HSV coordinate u_uv from r_uv
        # u = [(1-p)r]^{1/(1-p)}
        # Special case p ≈ 0: u = r (limit is well-defined)
        if p < 1e-10:
            u_uv = r_uv
            du_dr = 1.0
            one_minus_p = 1.0
        else:
            one_minus_p = 1.0 - p
            u_uv = ((one_minus_p * r_uv) ** (1.0 / one_minus_p))
            # du/dr = u^p
            du_dr = u_uv ** p

        # UV stabilization factor: [(1-p)·r_uv]^{-dγ}
        # For p=0: stab_factor = 1 (no stabilization needed)
        # This multiplies the raw propagator to give the UV-stabilized version
        if p < 1e-10:
            stab_factor = 1.0
        else:
            stab_factor = ((one_minus_p * r_uv) ** (-d * kg_gamma))
        
        # Coefficient for momentum correction: dγ/r_uv
        # Appears in: Π̃ = [(1-p)r]^{-dγ} · [Π - (dγ/r)·Φ]
        d_gamma = d * kg_gamma
        pi_correction_coeff = d_gamma / r_uv if r_uv > 1e-10 else 0.0

        # Normalization constant: 2^{β-1} Γ(β)
        normalization = (2.0 ** (beta - 1.0)) * scipy_gamma(beta)

        # Wave numbers using FFT ordering
        kx_freq = torch.fft.fftfreq(n_modes) * n_modes
        ky_freq = torch.fft.fftfreq(n_modes) * n_modes

        kx = 2.0 * math.pi * kx_freq / Lx
        ky = 2.0 * math.pi * ky_freq / Ly

        KX, KY = torch.meshgrid(kx, ky, indexing="ij")
        k_sq = KX ** 2 + KY ** 2
        k_mag = torch.sqrt(k_sq + regularization ** 2)

        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)
        self.register_buffer("KX", KX)
        self.register_buffer("KY", KY)
        self.register_buffer("k_sq", k_sq)
        self.register_buffer("k_mag", k_mag)

        # Store HSV-specific parameters
        self.register_buffer("u_uv", torch.tensor(u_uv, dtype=torch.float32))
        self.register_buffer("z_uv", torch.tensor(u_uv, dtype=torch.float32))  # Compatibility alias
        self.hsv_p = p
        self.hsv_beta = beta
        self.hsv_kg_gamma = kg_gamma

        # Precompute Bessel-based envelopes for each channel
        envelopes_phi = []
        envelopes_pi = []

        k_mag_np = k_mag.numpy()
        xi = k_mag_np * u_uv  # |k| · u_uv (argument to Bessel function)

        for c in range(self.n_channels):
            # HSV propagator is mass-independent at leading order in UV
            # (unlike AdS where mass determines the conformal dimension)
            # We use the universal HSV kernel for all channels

            # Raw propagator: κ = (|k|u)^β K_β(|k|u) / normalization
            # Handle small argument carefully
            xi_safe = np.maximum(xi, 1e-10)

            # Compute K_β(ξ) using scipy
            K_beta_vals = scipy_kv(beta, xi_safe)

            # Raw propagator: (|k|u)^β · K_β(|k|u) / normalization
            kappa_raw = (xi_safe ** beta) * K_beta_vals / normalization

            # Handle ξ → 0 limit: (ξ)^β K_β(ξ) → 2^{β-1} Γ(β) as ξ → 0
            # So κ → 1 as ξ → 0 (correctly normalized)
            kappa_raw = np.where(xi < 1e-10, 1.0, kappa_raw)

            # Raw momentum: dκ/dr = dκ/du · du/dr
            # Using d/dx[x^β K_β(x)] = -x^β K_{β-1}(x):
            # dκ/du = -|k| · (|k|u)^β K_{β-1}(|k|u) / normalization
            # dκ/dr = -|k| · (|k|u)^β K_{β-1}(|k|u) · du/dr / normalization

            # K_{β-1} (note: K_ν = K_{-ν}, so this works for β < 1 too)
            K_beta_minus_1_vals = scipy_kv(abs(beta - 1.0), xi_safe)

            # dκ/dr = -|k| · (|k|u)^β · K_{β-1}(|k|u) · du/dr / normalization
            dkappa_dr_raw = -k_mag_np * (xi_safe ** beta) * K_beta_minus_1_vals * du_dr / normalization

            # Handle ξ → 0 limit for momentum
            dkappa_dr_raw = np.where(xi < 1e-10, 0.0, dkappa_dr_raw)

            # =================================================================
            # Apply UV stabilization (code convention: p=0 flat, p→1 AdS)
            #
            # Field redefinition:
            #   Φ̃ = [(1-p)r]^{-dγ} · Φ
            #   Π̃ = [(1-p)r]^{-dγ} · [Π - (dγ/r)·Φ]
            #
            # For propagator at r_uv:
            #   κ̃ = stab_factor · κ
            #   π̃ = stab_factor · [dκ/dr - (dγ/r_uv)·κ]
            #
            # where stab_factor = [(1-p)·r_uv]^{-dγ}
            # =================================================================
            
            envelope_phi_np = stab_factor * kappa_raw
            envelope_pi_np = stab_factor * (dkappa_dr_raw - pi_correction_coeff * kappa_raw)

            # Normalize so zero-mode (k=0) has unit amplitude for phi
            norm_val = envelope_phi_np[0, 0]
            if abs(norm_val) > 1e-10:
                envelope_phi_np = envelope_phi_np / norm_val
                envelope_pi_np = envelope_pi_np / norm_val

            envelopes_phi.append(torch.tensor(envelope_phi_np, dtype=torch.float32))
            envelopes_pi.append(torch.tensor(envelope_pi_np, dtype=torch.float32))

        self.register_buffer("envelopes_phi", torch.stack(envelopes_phi, dim=0))
        self.register_buffer("envelopes_pi", torch.stack(envelopes_pi, dim=0))

        # Print diagnostic info
        print(f"[SPECTRAL-HSV] Planar HSV propagator with p={p:.4f}")
        print(f"[SPECTRAL-HSV] Bessel order β = (1 + (d-1)p)/2 = {beta:.4f}")
        print(f"[SPECTRAL-HSV] KG gamma γ = p/(1-p) = {kg_gamma:.4f}")
        print(f"[SPECTRAL-HSV] UV stabilization factor = [(1-p)·r_uv]^{{-dγ}} = {stab_factor:.6f}")
        print(f"[SPECTRAL-HSV] HSV coordinate u_uv = {u_uv:.6f} (from r_uv = {r_uv:.4f})")
        print(f"[SPECTRAL-HSV] Propagator: κ̃ = [(1-p)r]^{{-dγ}} · (|k|u)^β K_β(|k|u) / (2^{{β-1}} Γ(β))")
        if p < 0.01:
            print(f"[SPECTRAL-HSV] Near flat limit (p≈0): γ=0, no UV stabilization needed")
        elif p > 0.9:
            print(f"[SPECTRAL-HSV] Near AdS limit (p≈1): switches to AdS formulas at p≥0.95")

    def forward(self, points: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode points into spectral coefficients.

        Geometry-aware encoding:
        - PLANAR: Fourier transform with Bessel propagator
        - FLAT: Fourier with flat-space propagator (ablation)
        - HYPERSCALING_VIOLATING: Fourier with HSV propagator

        Args:
            points: (B, 2) source point coordinates

        Returns:
            phi_hat: (B, 2C, K, K) spectral coefficients (real, imag interleaved per channel)
            pi_hat: (B, 2C, K, K) momentum spectral coefficients
        """
        # ALL geometries use Fourier encoding for point cloud data
        # The geometry-specific physics is captured in:
        # 1. The envelopes (from propagator computation)
        # 2. The backbone dynamics (f(r) warp factor)
        return self._forward_fourier(points)

    def _forward_fourier(self, points: Tensor) -> Tuple[Tensor, Tensor]:
        """Fourier-based encoding for planar geometry."""
        B = points.shape[0]
        device = points.device
        dtype = points.dtype
        K = self.n_modes

        # Get coordinates relative to domain origin
        x = points[:, 0] - self.x_min.to(device)
        y = points[:, 1] - self.y_min.to(device)

        KX = self.KX.to(device=device, dtype=dtype)
        KY = self.KY.to(device=device, dtype=dtype)
        envelopes_phi = self.envelopes_phi.to(device=device, dtype=dtype)
        envelopes_pi = self.envelopes_pi.to(device=device, dtype=dtype)

        # Phase: e^{-i(k_x x + k_y y)}
        phase = x.view(B, 1, 1) * KX.unsqueeze(0) + y.view(B, 1, 1) * KY.unsqueeze(0)

        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        # Apply channel-specific envelopes
        phi_hat_list = []
        pi_hat_list = []

        for c in range(self.n_channels):
            env_phi = envelopes_phi[c]
            env_pi = envelopes_pi[c]

            phi_real = env_phi.unsqueeze(0) * cos_phase
            phi_imag = -env_phi.unsqueeze(0) * sin_phase

            pi_real = env_pi.unsqueeze(0) * cos_phase
            pi_imag = -env_pi.unsqueeze(0) * sin_phase

            phi_hat_list.append(phi_real)
            phi_hat_list.append(phi_imag)
            pi_hat_list.append(pi_real)
            pi_hat_list.append(pi_imag)

        phi_hat = torch.stack(phi_hat_list, dim=1)
        pi_hat = torch.stack(pi_hat_list, dim=1)

        return phi_hat, pi_hat

    def to_position_space(self, phi_hat: Tensor, grid_size: int = 64) -> Tensor:
        """
        Convert spectral coefficients to position space field.

        Args:
            phi_hat: (B, 2C, K, K) spectral coefficients
            grid_size: Output grid resolution

        Returns:
            phi: (B, C, H, W) position space field
        """
        B = phi_hat.shape[0]
        device = phi_hat.device
        dtype = phi_hat.dtype
        K = self.n_modes
        C = self.n_channels

        # Reshape: (B, 2C, K, K) -> (B, C, 2, K, K)
        phi_hat_ri = phi_hat.view(B, C, 2, K, K)

        # Complex tensor: (B, C, K, K)
        phi_hat_complex = torch.complex(phi_hat_ri[:, :, 0], phi_hat_ri[:, :, 1])

        # Zero-pad if needed
        if grid_size > K:
            phi_hat_padded = torch.zeros(
                B, C, grid_size, grid_size, dtype=phi_hat_complex.dtype, device=device
            )

            half_K = K // 2
            phi_hat_padded[:, :, :half_K, :half_K] = phi_hat_complex[
                :, :, :half_K, :half_K
            ]
            phi_hat_padded[:, :, :half_K, -half_K:] = phi_hat_complex[
                :, :, :half_K, -half_K:
            ]
            phi_hat_padded[:, :, -half_K:, :half_K] = phi_hat_complex[
                :, :, -half_K:, :half_K
            ]
            phi_hat_padded[:, :, -half_K:, -half_K:] = phi_hat_complex[
                :, :, -half_K:, -half_K:
            ]

            scale = (grid_size / K) ** 2
            phi_hat_padded = phi_hat_padded * scale
        else:
            phi_hat_padded = phi_hat_complex
            grid_size = K

        # Inverse FFT
        phi = torch.fft.ifft2(phi_hat_padded).real

        return phi

    def decode(self, phi_hat: Tensor) -> Tensor:
        """
        Decode spectral coefficients to position using phase gradient (Fourier shift theorem).

        For a point source at position x', the spectral representation is:
            φ̂(k) = E(|k|) · e^{-ik·x'}

        The position is encoded in the PHASE. We extract it via:
            φ̂(k+Δk) · φ̂*(k) = |E(k+Δk)||E(k)| · e^{-iΔk·x'}
            x' = -arg(...) / Δk

        We compute estimates per-channel to avoid phase cancellation, then average.

        Args:
            phi_hat: (B, 2C, K, K) spectral coefficients

        Returns:
            points: (B, 2) decoded boundary coordinates
        """
        B = phi_hat.shape[0]
        K = self.n_modes
        C = self.n_channels
        device = phi_hat.device
        dtype = phi_hat.dtype

        # Domain info
        Lx = self.Lx.to(device=device, dtype=dtype)
        Ly = self.Ly.to(device=device, dtype=dtype)
        x_min = self.x_min.to(device=device, dtype=dtype)
        y_min = self.y_min.to(device=device, dtype=dtype)

        # Wave number spacing
        dk_x = 2.0 * math.pi / Lx
        dk_y = 2.0 * math.pi / Ly

        # Reshape to complex: (B, 2C, K, K) -> (B, C, K, K) complex
        phi_hat_ri = phi_hat.view(B, C, 2, K, K)
        phi_complex = torch.complex(
            phi_hat_ri[:, :, 0], phi_hat_ri[:, :, 1]
        )  # (B, C, K, K)

        # ===== Create validity masks =====
        mask = torch.ones(K, K, device=device, dtype=dtype)

        # Exclude DC mode (k=0)
        mask[0, :] = 0
        mask[:, 0] = 0

        # Exclude wrapped edges
        mask[-1, :] = 0
        mask[:, -1] = 0

        # Exclude positive-to-negative frequency boundary
        half_K = K // 2
        if half_K > 0:
            mask[half_K - 1, :] = 0
            mask[:, half_K - 1] = 0
            mask[half_K, :] = 0
            mask[:, half_K] = 0

        # ===== Compute position estimates per channel =====
        x_weighted_sum = torch.zeros(B, device=device, dtype=dtype)
        y_weighted_sum = torch.zeros(B, device=device, dtype=dtype)
        x_weight_sum = torch.zeros(B, device=device, dtype=dtype)
        y_weight_sum = torch.zeros(B, device=device, dtype=dtype)

        for c in range(C):
            phi_c = phi_complex[:, c, :, :]  # (B, K, K)

            # X direction: shift along dim=1 (kx)
            phi_shift_x = torch.roll(phi_c, shifts=-1, dims=1)
            prod_x = phi_shift_x * phi_c.conj()

            # Y direction: shift along dim=2 (ky)
            phi_shift_y = torch.roll(phi_c, shifts=-1, dims=2)
            prod_y = phi_shift_y * phi_c.conj()

            # Phase differences → position estimates
            phase_diff_x = torch.angle(prod_x)
            phase_diff_y = torch.angle(prod_y)

            x_est = -phase_diff_x / dk_x  # (B, K, K)
            y_est = -phase_diff_y / dk_y

            # Weight by product magnitude (reliability of this estimate)
            weight_x = torch.abs(prod_x) * mask  # (B, K, K)
            weight_y = torch.abs(prod_y) * mask

            # Accumulate weighted estimates
            x_weighted_sum += (weight_x * x_est).sum(dim=(1, 2))
            y_weighted_sum += (weight_y * y_est).sum(dim=(1, 2))
            x_weight_sum += weight_x.sum(dim=(1, 2))
            y_weight_sum += weight_y.sum(dim=(1, 2))

        # Weighted average
        x_avg = x_weighted_sum / (x_weight_sum + 1e-10)
        y_avg = y_weighted_sum / (y_weight_sum + 1e-10)

        # Map from [-L/2, L/2] to [0, L] using modular arithmetic
        x_local = torch.remainder(x_avg, Lx)
        y_local = torch.remainder(y_avg, Ly)

        # Convert to domain coordinates
        x_out = x_local + x_min
        y_out = y_local + y_min

        return torch.stack([x_out, y_out], dim=-1)

    def decode_softargmax(self, phi_hat: Tensor, grid_size: int = 128) -> Tensor:
        """
        Alternative decoder using soft-argmax on position-space reconstruction.

        This method is robust to phase distortions but has grid resolution limits.
        Use higher grid_size for finer precision.

        Uses ADAPTIVE temperature to handle varying field statistics.

        Args:
            phi_hat: (B, 2C, K, K) spectral coefficients
            grid_size: Grid resolution for soft-argmax

        Returns:
            points: (B, 2) decoded coordinates
        """
        phi = self.to_position_space(phi_hat, grid_size=grid_size)

        B, C, H, W = phi.shape
        device = phi.device
        dtype = phi.dtype

        # Sum across channels for intensity
        intensity = phi.sum(dim=1)  # (B, H, W)
        intensity_flat = intensity.view(B, -1)  # (B, H*W)

        # ADAPTIVE TEMPERATURE: Normalize logits to have target std for good softmax behavior
        # This prevents both:
        # - Grid artifacts from too-peaked softmax (when field has large variance)
        # - Uniform sampling (when field has small variance)

        # Compute per-batch mean and std
        field_mean = intensity_flat.mean(dim=-1, keepdim=True)  # (B, 1)
        field_std = intensity_flat.std(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, 1)

        # Standardize the field (zero mean, unit std)
        standardized = (intensity_flat - field_mean) / field_std  # (B, H*W)

        # Target std for logits: controls peakedness of softmax
        # Higher = more peaked (sharper), Lower = smoother
        # Value of 3-5 gives good balance between sharp peaks and smooth sampling
        target_logit_std = 4.0
        logits = standardized * target_logit_std

        # Softmax to get weights
        weights = F.softmax(logits, dim=-1).view(B, H, W)

        Lx = self.Lx.to(device=device, dtype=dtype)
        Ly = self.Ly.to(device=device, dtype=dtype)
        x_min = self.x_min.to(device=device, dtype=dtype)
        y_min = self.y_min.to(device=device, dtype=dtype)

        # FIXED: In FFT convention, first dimension (H) corresponds to X,
        # second dimension (W) corresponds to Y
        x_local = torch.arange(H, device=device, dtype=dtype) * Lx / H
        y_local = torch.arange(W, device=device, dtype=dtype) * Ly / W

        XX, YY = torch.meshgrid(x_local, y_local, indexing="ij")

        x_out = (weights * XX).sum(dim=(1, 2)) + x_min
        y_out = (weights * YY).sum(dim=(1, 2)) + y_min

        return torch.stack([x_out, y_out], dim=-1)


# =============================================================================
# Spectral Laplacian (Diagonal in Fourier Space)
# =============================================================================


class SpectralLaplacian(nn.Module):
    """
    Laplacian operator in spectral space.

    This is DIAGONAL: Δ φ̂_k = -|k|² φ̂_k

    No FFT needed - just multiply by precomputed -k²!

    Attributes
    ----------
    n_modes : int
        Number of spectral modes
    n_channels : int
        Number of field channels
    minus_k_sq : Tensor
        Precomputed -|k|² values

    Example
    -------
    >>> laplacian = SpectralLaplacian(n_modes=16, n_channels=2, k_sq=encoder.k_sq)
    >>> lap_phi = laplacian.apply_minus_laplacian(phi_hat)
    """

    def __init__(self, n_modes: int, n_channels: int, k_sq: Tensor) -> None:
        """
        Initialize spectral Laplacian.

        Args:
            n_modes: Number of spectral modes
            n_channels: Number of field channels
            k_sq: (K, K) tensor of |k|² values
        """
        super().__init__()
        self.n_modes = n_modes
        self.n_channels = n_channels
        # k_sq: (K, K)
        self.register_buffer("minus_k_sq", -k_sq)

    def apply_minus_laplacian(self, phi_hat: Tensor) -> Tensor:
        """
        Apply (-Δ) in spectral space.

        Args:
            phi_hat: (B, 2C, K, K) spectral coefficients

        Returns:
            lap_phi_hat: (B, 2C, K, K) = |k|² φ̂
        """
        # -Δ φ = |k|² φ in Fourier space
        # So (-Δ) φ̂ = |k|² φ̂
        k_sq = (-self.minus_k_sq).to(
            device=phi_hat.device, dtype=phi_hat.dtype
        )  # (K, K)
        return phi_hat * k_sq.unsqueeze(0).unsqueeze(0)  # (B, 2C, K, K)

    # NOTE: diag_eigs intentionally NOT implemented for SpectralLaplacian
    # This matches the old code behavior where spectral encoding uses
    # uniform variance IR prior (fallback path in sample_ir_prior)


# =============================================================================
# Spectral Holographic Codec (Combined Encoder + Decoder + Laplacian)
# =============================================================================


@register_encoder("spectral_codec")
class SpectralHolographicCodec(nn.Module):
    """
    Geometry-aware spectral holographic encoder-decoder.

    This is the RECOMMENDED approach for large grids:
    - O(K²) memory instead of O(H×W)
    - O(K²) Laplacian instead of O(HW log(HW))
    - Exact AdS/CFT physics with geometry-appropriate spectral basis
    - Can decode to arbitrary resolution

    Supports slice geometries:
    - PLANAR: Fourier basis with Bessel propagator
    - FLAT: Flat-space propagator (ablation)
    - HYPERSCALING_VIOLATING: HSV propagator

    Supports multiple decoding methods:
    - 'density': CFT-faithful density sampling (RECOMMENDED for generation)
    - 'softargmax': Robust soft-argmax on grid
    - 'phase': Phase-gradient based (sub-grid precision)

    Example
    -------
    >>> codec = SpectralHolographicCodec(
    ...     d=2, deltas=[1.5, 2.0], n_modes=16,
    ...     geometry=SliceGeometry.PLANAR
    ... )
    >>> phi_hat, pi_hat = codec.encode(points)
    >>> points_decoded = codec.decode(phi_hat, method='density')
    >>> laplacian = codec.get_laplacian()
    """

    def __init__(
        self,
        d: int,
        deltas: Sequence[float],
        n_modes: int = 16,
        r_uv: float = 1.0,
        domain: Tuple[float, float, float, float] = (-4.0, 4.0, -4.0, 4.0),
        regularization: float = 1e-2,
        decode_resolution: int = 64,
        # Density sampler settings
        density_grid_size: int = 256,
        density_temperature: float = 0.1,
        # Geometry
        geometry: SliceGeometry = SliceGeometry.PLANAR,
        hsv_p: Optional[float] = None,
        hsv_propagator_mode: str = "bessel",
    ) -> None:
        """
        Initialize spectral holographic codec.

        Args:
            d: Boundary dimension
            deltas: Conformal dimensions per channel
            n_modes: Spectral modes per dimension
            r_uv: UV boundary radius
            domain: (x_min, x_max, y_min, y_max) bounds
            regularization: Numerical stability parameter
            decode_resolution: Default decode grid resolution
            density_grid_size: Grid size for density sampling
            density_temperature: Temperature for density sampling
            geometry: Slice geometry type
            hsv_p: HSV parameter p (only used for HSV geometries)
            hsv_propagator_mode: "bessel" for closed-form conformal product,
                                 "emd" for full EMD ODE integration
        """
        super().__init__()
        self.geometry = geometry

        print(f"[SPECTRAL-CODEC] Geometry: {geometry.value}")

        self.encoder = SpectralHolographicEncoder(
            d=d,
            deltas=deltas,
            n_modes=n_modes,
            r_uv=r_uv,
            domain=domain,
            regularization=regularization,
            geometry=geometry,
            hsv_p=hsv_p,
            hsv_propagator_mode=hsv_propagator_mode,
        )
        self.d = d
        self.n_channels = len(deltas)
        self.n_modes = n_modes
        self.r_uv = r_uv
        self.domain = domain
        self.decode_resolution = decode_resolution

        # Create spectral Laplacian with geometry-appropriate eigenvalues
        self.laplacian = SpectralLaplacian(
            n_modes=n_modes,
            n_channels=len(deltas),
            k_sq=self.encoder.k_sq,
        )

        deltas_t = torch.tensor(list(deltas), dtype=torch.float32)
        self.register_buffer("deltas", deltas_t)

        # Store density sampler settings (created lazily)
        self._density_grid_size = density_grid_size
        self._density_temperature = density_temperature
        self._density_sampler: Optional["FieldDensitySampler"] = None

    @property
    def density_sampler(self) -> "FieldDensitySampler":
        """Lazy creation of density sampler."""
        if self._density_sampler is None:
            self._density_sampler = FieldDensitySampler(
                encoder=self.encoder,
                grid_size=self._density_grid_size,
                temperature=self._density_temperature,
                subpixel_jitter=True,
            )
        return self._density_sampler

    def encode(self, points: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode points to spectral coefficients.

        Args:
            points: (B, 2) boundary point coordinates

        Returns:
            phi_hat: (B, 2C, K, K) field spectral coefficients
            pi_hat: (B, 2C, K, K) momentum spectral coefficients
        """
        return self.encoder(points)

    def decode(self, phi_hat: Tensor, method: str = "softargmax") -> Tensor:
        """
        Decode spectral coefficients to points (single point per sample).

        Args:
            phi_hat: (B, 2C, K, K) spectral coefficients
            method: 'softargmax' (robust), 'phase' (sub-grid), or 'density' (sample one)

        Returns:
            points: (B, 2) decoded coordinates
        """
        if method == "phase":
            return self.encoder.decode(phi_hat)
        elif method == "density":
            return self.density_sampler.sample_single(phi_hat)
        else:
            return self.encoder.decode_softargmax(phi_hat)

    def sample_points(
        self,
        phi_hat: Tensor,
        n_samples: int,
        temperature: Optional[float] = None,
        deterministic: bool = False,
    ) -> Tensor:
        """
        Sample multiple points from the field density (CFT-faithful).

        This is the RECOMMENDED method for generation.

        Args:
            phi_hat: (B, 2C, K, K) spectral coefficients
            n_samples: Number of points to sample per batch element
            temperature: Override temperature (None = use default)
            deterministic: If True, return top-k positions

        Returns:
            points: (B, n_samples, 2) sampled coordinates
        """
        if temperature is not None:
            # Create temporary sampler with different temperature
            sampler = FieldDensitySampler(
                encoder=self.encoder,
                grid_size=self._density_grid_size,
                temperature=temperature,
                subpixel_jitter=True,
            )
            return sampler.sample(phi_hat, n_samples, deterministic)
        else:
            return self.density_sampler.sample(phi_hat, n_samples, deterministic)

    def get_laplacian(self) -> SpectralLaplacian:
        """Return the spectral Laplacian operator."""
        return self.laplacian

    def get_density(
        self, phi_hat: Tensor, grid_size: Optional[int] = None
    ) -> Tensor:
        """
        Get the density field for visualization.

        Args:
            phi_hat: (B, 2C, K, K) spectral coefficients
            grid_size: Override grid size

        Returns:
            density: (B, H, W) normalized density
        """
        gs = grid_size or self._density_grid_size
        phi = self.encoder.to_position_space(phi_hat, grid_size=gs)
        return self.density_sampler.field_to_density(phi)


# =============================================================================
# Field Density Sampler (CFT-Faithful Readout)
# =============================================================================


class FieldDensitySampler(nn.Module):
    """
    CFT-faithful readout: interpret the boundary field as a probability density
    and SAMPLE points from it.

    In QFT/CFT, the field value determines the probability of finding particles
    at each location. This is the natural way to go from fields to point clouds.

    The field Φ(x) defines a density:
        p(x) ∝ exp(Φ(x) / T)    [Boltzmann/energy interpretation]
    or
        p(x) ∝ softmax(Φ(x) / T) [numerically stable version]

    Why this is better than point decoders:
        - No information bottleneck (field → distribution → samples)
        - Naturally produces multi-modal distributions
        - Sharp boundaries emerge from the density
        - Sampling is stochastic, matching true generative modeling
        - Fully CFT-faithful interpretation

    Attributes
    ----------
    encoder : SpectralHolographicEncoder
        Encoder for spectral → position space conversion
    grid_size : int
        Resolution for density evaluation
    temperature : float
        Controls sharpness (lower = sharper peaks)
    subpixel_jitter : bool
        Add uniform noise for continuous output

    Example
    -------
    >>> sampler = FieldDensitySampler(encoder, grid_size=256)
    >>> points = sampler.sample(phi_hat, n_samples=2000)
    """

    def __init__(
        self,
        encoder: SpectralHolographicEncoder,
        grid_size: int = 256,
        temperature: float = 0.1,
        subpixel_jitter: bool = True,
        langevin_steps: int = 0,
        langevin_step_size: float = 0.01,
    ) -> None:
        """
        Initialize density sampler.

        Args:
            encoder: SpectralHolographicEncoder to convert spectral → position space
            grid_size: Resolution for density evaluation (higher = finer details)
            temperature: Controls sharpness of density (lower = sharper peaks)
            subpixel_jitter: Add uniform noise within each pixel for continuous output
            langevin_steps: Optional Langevin refinement steps (0 = disabled)
            langevin_step_size: Step size for Langevin dynamics
        """
        super().__init__()
        self.encoder = encoder
        self.grid_size = grid_size
        self.temperature = temperature
        self.subpixel_jitter = subpixel_jitter
        self.langevin_steps = langevin_steps
        self.langevin_step_size = langevin_step_size

        # Cache domain info
        self.register_buffer("Lx", encoder.Lx.clone())
        self.register_buffer("Ly", encoder.Ly.clone())
        self.register_buffer("x_min", encoder.x_min.clone())
        self.register_buffer("y_min", encoder.y_min.clone())

    def field_to_density(self, phi: Tensor) -> Tensor:
        """
        Convert field to normalized probability density.

        Uses ADAPTIVE temperature scaling to handle varying field statistics:
        - Standardizes field to have consistent scale before softmax
        - Prevents both grid artifacts (too peaked) and uniform sampling (too flat)

        Args:
            phi: (B, C, H, W) field in position space

        Returns:
            density: (B, H, W) normalized probability density
        """
        B, C, H, W = phi.shape

        # Sum across channels to get scalar field
        # This combines all conformal dimensions into one density
        scalar_field = phi.sum(dim=1)  # (B, H, W)
        scalar_flat = scalar_field.view(B, -1)  # (B, H*W)

        # ADAPTIVE TEMPERATURE: Normalize logits to have target std for good softmax behavior
        # This prevents both:
        # - Grid artifacts from too-peaked softmax (when field has large variance)
        # - Uniform sampling (when field has small variance)

        # Compute per-batch mean and std
        field_mean = scalar_flat.mean(dim=-1, keepdim=True)  # (B, 1)
        field_std = scalar_flat.std(dim=-1, keepdim=True).clamp(min=1e-6)  # (B, 1)

        # Standardize the field (zero mean, unit std)
        standardized = (scalar_flat - field_mean) / field_std  # (B, H*W)

        # Target std for logits: controls peakedness of softmax
        # Higher = more peaked (sharper), Lower = smoother
        # For density sampling with multinomial, we want moderate peakedness
        # Value around 3-5 gives good balance
        target_logit_std = 4.0 / self.temperature  # Allow temperature to influence
        target_logit_std = min(max(target_logit_std, 2.0), 10.0)  # Clamp to reasonable range
        logits = standardized * target_logit_std

        # Softmax to get normalized density
        density_flat = F.softmax(logits, dim=-1)
        density = density_flat.view(B, H, W)

        return density

    def sample_from_density(
        self,
        density: Tensor,
        n_samples: int,
        deterministic: bool = False,
    ) -> Tensor:
        """
        Sample points from a 2D density.

        Args:
            density: (B, H, W) normalized probability density
            n_samples: Number of points to sample per batch element
            deterministic: If True, use top-k instead of random sampling

        Returns:
            points: (B, n_samples, 2) sampled coordinates in domain space
        """
        B, H, W = density.shape
        device = density.device
        dtype = density.dtype

        # Flatten density for sampling
        density_flat = density.view(B, -1)  # (B, H*W)

        if deterministic:
            # Take top-k most likely positions
            _, indices = torch.topk(density_flat, n_samples, dim=-1)
        else:
            # Multinomial sampling from the density
            # This is the core of the CFT interpretation!
            indices = torch.multinomial(
                density_flat, n_samples, replacement=True
            )  # (B, n_samples)

        # Convert flat indices to 2D grid coordinates
        # FIXED: In FFT convention with indexing='ij':
        #   - First dimension (H) corresponds to X
        #   - Second dimension (W) corresponds to Y
        # flat_index = x_idx * W + y_idx
        ix = indices // W  # X index (first dim = rows)
        iy = indices % W  # Y index (second dim = columns)

        # Convert to continuous coordinates
        # Grid spacing
        dx = self.Lx / H  # X spacing = X-range / X-size
        dy = self.Ly / W  # Y spacing = Y-range / Y-size

        # Base coordinates (pixel centers)
        x_base = ix.float() * dx
        y_base = iy.float() * dy

        if self.subpixel_jitter and not deterministic:
            # Add uniform jitter within each pixel
            # This removes grid artifacts and gives continuous positions
            x_jitter = torch.rand_like(x_base) * dx
            y_jitter = torch.rand_like(y_base) * dy
            x_local = x_base + x_jitter
            y_local = y_base + y_jitter
        else:
            # Use pixel centers
            x_local = x_base + dx / 2
            y_local = y_base + dy / 2

        # Convert to domain coordinates
        x = x_local + self.x_min
        y = y_local + self.y_min

        # Stack to (B, n_samples, 2)
        points = torch.stack([x, y], dim=-1)

        return points

    def langevin_refinement(
        self,
        points: Tensor,
        phi: Tensor,
    ) -> Tensor:
        """
        Refine sample positions using Langevin dynamics on the density.

        This moves points toward higher density regions while maintaining
        stochasticity, resulting in better coverage of the modes.

        Args:
            points: (B, n_samples, 2) initial positions
            phi: (B, C, H, W) field for gradient computation

        Returns:
            refined_points: (B, n_samples, 2) refined positions
        """
        if self.langevin_steps == 0:
            return points

        B, N, _ = points.shape
        _, C, H, W = phi.shape
        device = points.device
        dtype = points.dtype

        # Get domain info
        Lx = self.Lx.to(device)
        Ly = self.Ly.to(device)
        x_min = self.x_min.to(device)
        y_min = self.y_min.to(device)

        # Convert points to normalized coordinates [0, 1] for grid_sample
        points_normalized = points.clone()
        points_normalized[..., 0] = (points[..., 0] - x_min) / Lx * 2 - 1  # [-1, 1]
        points_normalized[..., 1] = (points[..., 1] - y_min) / Ly * 2 - 1

        # Sum field across channels
        scalar_field = phi.sum(dim=1, keepdim=True) / self.temperature  # (B, 1, H, W)

        current_points = points.clone()

        for _ in range(self.langevin_steps):
            # Enable gradients for gradient computation
            current_points.requires_grad_(True)

            # Convert to grid_sample format
            grid = torch.stack(
                [
                    (current_points[..., 0] - x_min) / Lx * 2 - 1,
                    (current_points[..., 1] - y_min) / Ly * 2 - 1,
                ],
                dim=-1,
            )  # (B, N, 2)
            grid = grid.unsqueeze(2)  # (B, N, 1, 2)

            # Sample field values at current positions
            field_vals = F.grid_sample(
                scalar_field,
                grid,
                mode="bilinear",
                align_corners=True,
                padding_mode="border",
            )  # (B, 1, N, 1)
            field_vals = field_vals.squeeze(1).squeeze(-1)  # (B, N)

            # Compute gradient of log-density (= field / T)
            log_density = field_vals.sum()
            log_density.backward()

            grad = current_points.grad  # (B, N, 2)

            # Langevin update: x += step_size * grad + sqrt(2*step_size) * noise
            with torch.no_grad():
                noise = torch.randn_like(current_points)
                current_points = (
                    current_points
                    + self.langevin_step_size * grad
                    + math.sqrt(2 * self.langevin_step_size) * noise * 0.1
                )

                # Clamp to domain
                current_points[..., 0].clamp_(x_min, x_min + Lx)
                current_points[..., 1].clamp_(y_min, y_min + Ly)

            current_points = current_points.detach()

        return current_points

    def sample(
        self,
        phi_hat: Tensor,
        n_samples: int = 1,
        deterministic: bool = False,
        return_density: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Sample points from the density defined by spectral coefficients.

        This is the main method for CFT-faithful point generation.

        Args:
            phi_hat: (B, 2C, K, K) spectral coefficients
            n_samples: Number of points to sample per batch element
            deterministic: If True, return top-k positions instead of sampling
            return_density: If True, also return the density field

        Returns:
            points: (B, n_samples, 2) sampled coordinates
            density: (B, H, W) density field (if return_density=True)
        """
        # 1. Convert spectral coefficients to position-space field
        phi = self.encoder.to_position_space(phi_hat, grid_size=self.grid_size)

        # 2. Convert field to normalized density
        density = self.field_to_density(phi)

        # 3. Sample from the density
        points = self.sample_from_density(density, n_samples, deterministic)

        # 4. Optional Langevin refinement
        if self.langevin_steps > 0 and not deterministic:
            points = self.langevin_refinement(points, phi)

        if return_density:
            return points, density
        return points

    def sample_single(self, phi_hat: Tensor) -> Tensor:
        """
        Sample a single point per batch element.

        This maintains API compatibility with the decode() interface.

        Args:
            phi_hat: (B, 2C, K, K) spectral coefficients

        Returns:
            points: (B, 2) single sampled point per batch element
        """
        points = self.sample(phi_hat, n_samples=1, deterministic=False)
        return points.squeeze(1)  # (B, 2)


# =============================================================================
# Grid-Free Holographic Point Encoder
# =============================================================================


@register_encoder("point_holographic")
class HolographicPointEncoder(nn.Module):
    """
    Grid-free holographic encoding for point data.

    This keeps points as coordinates but adds holographic physics:
    - Φ̃ = point coordinates (preserves the data!)
    - Π̃ = analytically computed from bulk-boundary propagator derivative
    - Laplacian effect computed analytically and added to backbone

    The key insight: we DON'T need to create a grid field. The point coordinates
    ARE the boundary data. The holographic propagator tells us HOW this data
    creates a bulk field, which affects the ODE evolution.

    Memory: O(C) per sample (same as no-encoding)
    Physics: Includes holographic Laplacian effects

    Attributes
    ----------
    d : int
        Boundary dimension
    n_channels : int
        Number of field channels
    r_uv : float
        UV boundary radius
    C_delta : Tensor
        Normalization constants C_Δ
    pi_to_phi_ratio : Tensor
        Momentum/field ratio from propagator

    Example
    -------
    >>> encoder = HolographicPointEncoder(d=2, deltas=[1.5, 2.0], r_uv=1.0)
    >>> phi_tilde, pi_tilde, lap_coeff = encoder(points)
    """

    def __init__(
        self,
        d: int,
        deltas: Sequence[float],
        r_uv: float,
        regularization: float = 1e-2,
    ) -> None:
        """
        Initialize grid-free holographic encoder.

        Args:
            d: Boundary dimension (2 for 2D points)
            deltas: Conformal dimensions Δ_i for each channel
            r_uv: UV radius where lift occurs
            regularization: Small ε for Laplacian regularization
        """
        super().__init__()
        self.d = d
        self.n_channels = len(deltas)
        self.r_uv = r_uv
        self.regularization = regularization

        deltas_t = torch.tensor(list(deltas), dtype=torch.float32)
        self.register_buffer("deltas", deltas_t)

        # Normalization C_Δ = Γ(Δ) / (π^{d/2} Γ(Δ - d/2))
        log_C = (
            torch.lgamma(deltas_t)
            - (d / 2.0) * math.log(math.pi)
            - torch.lgamma(deltas_t - d / 2.0)
        )
        C_delta = torch.exp(log_C)
        self.register_buffer("C_delta", C_delta)

        # z² = e^{-2r_UV}
        z_sq_uv = math.exp(-2.0 * r_uv)
        self.register_buffer("z_sq_uv", torch.tensor(z_sq_uv, dtype=torch.float32))

        # Precompute scale factors for pi_tilde
        # At UV, the propagator derivative w.r.t. r gives the momentum
        # Π̃ ~ 2Δ z² / (z² + ε)^{Δ+1} * C_Δ
        # We use this to scale the momentum relative to the field
        denom = z_sq_uv + regularization
        pi_scale = 2.0 * deltas_t * z_sq_uv * C_delta / (denom ** (deltas_t + 1.0))
        phi_scale = C_delta / (denom ** deltas_t)
        # Ratio gives us how to scale pi relative to phi
        self.register_buffer("pi_to_phi_ratio", pi_scale / (phi_scale + 1e-10))

    def forward(self, points: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute holographic encoding for points.

        Args:
            points: (B, 2) source point coordinates (this IS the data)

        Returns:
            phi_tilde: (B, C) - the point coordinates (C should equal 2 for 2D)
            pi_tilde: (B, C) - momentum scaled by holographic ratio
            laplacian_coeff: (B, C) - effective Laplacian term coefficient
        """
        B = points.shape[0]
        device = points.device
        dtype = points.dtype

        # Φ̃ = the point coordinates themselves
        # For 2D points with C=2 channels, this is just the identity
        phi_tilde = points  # (B, 2)

        # Ensure shape matches n_channels
        if phi_tilde.shape[1] != self.n_channels:
            raise ValueError(
                f"Point dimension {phi_tilde.shape[1]} doesn't match n_channels {self.n_channels}. "
                f"For 2D points, use deltas with 2 elements."
            )

        # Π̃ = scaled version of Φ̃ based on holographic propagator derivative
        # This encodes that ∂_r Φ̃ ~ (holographic ratio) × Φ̃ at UV
        pi_to_phi = self.pi_to_phi_ratio.to(device=device, dtype=dtype)
        pi_tilde = phi_tilde * pi_to_phi.unsqueeze(0)  # (B, C)

        # Laplacian coefficient: how much the Laplacian contributes
        # For a propagator, Δ K ~ -2Δd / (z² + ε) × K
        # This gives an effective "mass-like" term from the Laplacian
        z_sq = self.z_sq_uv.to(device=device, dtype=dtype)
        deltas = self.deltas.to(device=device, dtype=dtype)
        eps = self.regularization

        # Laplacian coefficient per channel
        lap_coeff = -2.0 * deltas * self.d / (z_sq + eps)  # (C,)
        laplacian_coeff = lap_coeff.unsqueeze(0).expand(B, -1)  # (B, C)

        return phi_tilde.clone(), pi_tilde, laplacian_coeff


class HolographicPointCodec(nn.Module):
    """
    Grid-free holographic codec for point data.

    This provides holographic physics without memory overhead:
    - Φ̃ = point coordinates (data preserved)
    - Π̃ = holographically scaled momentum
    - Laplacian effect included analytically

    Memory: O(C) per sample (same as no-encoding!)

    Example
    -------
    >>> codec = HolographicPointCodec(d=2, deltas=[1.5, 2.0], r_uv=1.0)
    >>> phi_tilde, pi_tilde, lap_coeff = codec.encode(points)
    >>> points_decoded = codec.decode(phi_tilde)
    """

    def __init__(
        self,
        d: int,
        deltas: Sequence[float],
        r_uv: float = 1.0,
        regularization: float = 1e-2,
    ) -> None:
        """
        Initialize grid-free holographic codec.

        Args:
            d: Boundary dimension
            deltas: Conformal dimensions per channel
            r_uv: UV boundary radius
            regularization: Numerical stability parameter
        """
        super().__init__()
        self.encoder = HolographicPointEncoder(
            d=d,
            deltas=deltas,
            r_uv=r_uv,
            regularization=regularization,
        )
        self.d = d
        self.n_channels = len(deltas)
        self.r_uv = r_uv

        deltas_t = torch.tensor(list(deltas), dtype=torch.float32)
        self.register_buffer("deltas", deltas_t)

    def encode(self, points: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Encode points with holographic structure.

        Args:
            points: (B, 2) boundary point coordinates

        Returns:
            phi_tilde: (B, C) point coordinates
            pi_tilde: (B, C) holographic momentum
            laplacian_coeff: (B, C) Laplacian term coefficients
        """
        return self.encoder(points)

    def decode(self, phi_tilde: Tensor) -> Tensor:
        """
        Decode back to points.

        For grid-free, Φ̃ IS the point coordinates!

        Args:
            phi_tilde: (B, C) field state

        Returns:
            points: (B, C) decoded coordinates (same as input)
        """
        return phi_tilde

    def get_laplacian_term(
        self,
        phi_tilde: Tensor,
        r: Union[float, Tensor],
    ) -> Tensor:
        """
        Compute the effective Laplacian term for the backbone.

        This approximates -(1/f²) Δ_ĝ Φ̃ using the holographic propagator.

        Args:
            phi_tilde: (B, C) current field state
            r: Current radius

        Returns:
            lap_term: (B, C) contribution to dΠ̃/dr
        """
        device = phi_tilde.device
        dtype = phi_tilde.dtype

        # z² = e^{-2r}
        if isinstance(r, Tensor):
            z_sq = torch.exp(-2.0 * r).to(device=device, dtype=dtype)
        else:
            z_sq = math.exp(-2.0 * r)

        deltas = self.deltas.to(device=device, dtype=dtype)
        eps = self.encoder.regularization

        # Effective Laplacian: -2Δd / (z² + ε) × Φ̃
        # This captures how the propagator's Laplacian scales
        scale = -2.0 * deltas * self.d / (z_sq + eps)  # (C,)
        if isinstance(scale, Tensor):
            scale = scale.unsqueeze(0)  # (1, C)

        return scale * phi_tilde


# =============================================================================
# Grid-Based Holographic Point-Field Encoder
# =============================================================================


@register_encoder("point_field_holographic")
class HolographicPointFieldEncoder(nn.Module):
    """
    Encode points to fields using the AdS/CFT bulk-to-boundary propagator.

    This is the EXACT holographic dictionary for how boundary point sources
    create bulk field configurations.

    The bulk-to-boundary propagator is:
        K_Δ(z, x; x') = C_Δ (z / (z² + |x - x'|²))^Δ

    where:
        - z = e^{-r} is the Poincaré coordinate (z→0 is UV/boundary)
        - C_Δ = Γ(Δ) / (π^{d/2} Γ(Δ - d/2)) is the normalization
        - Δ is the conformal dimension

    In UV-stabilized variables (Φ̃ = e^{Δr} Φ):
        K̃_Δ(r, x; x') = C_Δ / (e^{-2r} + |x - x'|²)^Δ

    The momentum Π̃ is computed analytically:
        Π̃ = 2Δ e^{-2r} C_Δ / (e^{-2r} + |x - x'|²)^{Δ+1}

    Key properties:
    - At UV (r→∞): K̃ becomes peaked at source → recovers point data
    - At IR (r→-∞): K̃ spreads uniformly → holographic RG coarse-graining
    - Satisfies Klein-Gordon equation in the bulk
    - The Laplacian Δ_ĝ acts nontrivially on the field!

    Attributes
    ----------
    d : int
        Boundary dimension
    n_channels : int
        Number of field channels
    grid_size : Tuple[int, int]
        Spatial grid resolution
    domain : Tuple[float, ...]
        Domain bounds
    r_uv : float
        UV boundary radius
    grid_x, grid_y : Tensor
        Coordinate grids

    Example
    -------
    >>> encoder = HolographicPointFieldEncoder(
    ...     d=2, deltas=[1.5, 2.0],
    ...     grid_size=(32, 32), domain=(-4, 4, -4, 4), r_uv=1.0
    ... )
    >>> phi_tilde, pi_tilde = encoder(points)
    """

    def __init__(
        self,
        d: int,
        deltas: Sequence[float],
        grid_size: Tuple[int, int],
        domain: Tuple[float, float, float, float],
        r_uv: float,
        regularization: float = 1e-6,
        normalize_integral: bool = True,
    ) -> None:
        """
        Initialize grid-based holographic encoder.

        Args:
            d: Boundary dimension (2 for standard images/points)
            deltas: Conformal dimensions Δ_i for each channel
            grid_size: (H, W) spatial resolution of output field
            domain: (x_min, x_max, y_min, y_max) coordinate bounds
            r_uv: UV radius where lift occurs (z_UV = e^{-r_UV})
            regularization: Small ε to avoid division by zero at source
            normalize_integral: If True, normalize so field integrates to ~1
        """
        super().__init__()
        self.d = d
        self.n_channels = len(deltas)
        self.grid_size = grid_size
        self.domain = domain
        self.r_uv = r_uv
        self.regularization = regularization
        self.normalize_integral = normalize_integral

        # Store conformal dimensions
        deltas_t = torch.tensor(list(deltas), dtype=torch.float32)
        self.register_buffer("deltas", deltas_t)

        # Compute normalization constants C_Δ = Γ(Δ) / (π^{d/2} Γ(Δ - d/2))
        # Using log-gamma for numerical stability
        log_C = (
            torch.lgamma(deltas_t)
            - (d / 2.0) * math.log(math.pi)
            - torch.lgamma(deltas_t - d / 2.0)
        )
        C_delta = torch.exp(log_C)
        self.register_buffer("C_delta", C_delta)

        # Precompute z² = e^{-2r_UV}
        z_sq_uv = math.exp(-2.0 * r_uv)
        self.register_buffer("z_sq_uv", torch.tensor(z_sq_uv, dtype=torch.float32))

        # Create coordinate grid
        H, W = grid_size
        x_min, x_max, y_min, y_max = domain

        # Grid spacing for normalization
        dx = (x_max - x_min) / W
        dy = (y_max - y_min) / H
        self.register_buffer("grid_area", torch.tensor(dx * dy, dtype=torch.float32))

        # Coordinate arrays
        u = torch.linspace(x_min + dx / 2, x_max - dx / 2, W)  # Cell centers
        v = torch.linspace(y_min + dy / 2, y_max - dy / 2, H)
        V, U = torch.meshgrid(v, u, indexing="ij")

        self.register_buffer("grid_x", U)  # (H, W)
        self.register_buffer("grid_y", V)  # (H, W)

        # Precompute per-channel normalization factors
        # These ensure the field integrates to approximately 1
        if normalize_integral:
            # For a propagator centered at origin with regularization ε:
            # ∫ C_Δ / (z² + ρ² + ε)^Δ dρ ≈ C_Δ π / ((Δ-1)(z²+ε)^{Δ-1}) for large domain
            # We'll compute normalization empirically for the actual grid
            norm_factors = []
            for c in range(self.n_channels):
                delta = float(deltas_t[c])
                C = float(C_delta[c])
                # Propagator at grid center
                rho_sq = U ** 2 + V ** 2
                denom = z_sq_uv + rho_sq + regularization
                K_test = C / (denom ** delta)
                integral = K_test.sum() * dx * dy
                norm_factors.append(1.0 / (integral + 1e-10))
            self.register_buffer(
                "norm_factors", torch.tensor(norm_factors, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "norm_factors", torch.ones(self.n_channels, dtype=torch.float32)
            )

    def compute_propagator(
        self,
        points: Tensor,
        r: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute UV-stabilized propagator and its r-derivative at given points.

        Args:
            points: (B, 2) source point coordinates (x, y)
            r: Optional radius (defaults to r_uv)

        Returns:
            phi_tilde: (B, C, H, W) - UV-stabilized field
            pi_tilde: (B, C, H, W) - UV-stabilized momentum (analytic)
        """
        B = points.shape[0]
        H, W = self.grid_size
        device = points.device
        dtype = points.dtype

        # Source coordinates
        px = points[:, 0].view(B, 1, 1)  # (B, 1, 1)
        py = points[:, 1].view(B, 1, 1)  # (B, 1, 1)

        # Grid coordinates
        gx = self.grid_x.to(device=device, dtype=dtype)  # (H, W)
        gy = self.grid_y.to(device=device, dtype=dtype)  # (H, W)

        # Squared distance from each grid point to source: ρ² = |x - x'|²
        rho_sq = (gx - px) ** 2 + (gy - py) ** 2  # (B, H, W)

        # z² = e^{-2r}
        if r is not None:
            z_sq = torch.exp(-2.0 * r.to(device=device, dtype=dtype))
        else:
            z_sq = self.z_sq_uv.to(device=device, dtype=dtype)

        # Ensure z_sq is a scalar
        if isinstance(z_sq, Tensor) and z_sq.numel() > 1:
            z_sq = z_sq.view(-1, 1, 1)  # (B, 1, 1) for batch-varying r

        # Denominator: e^{-2r} + ρ² + ε
        denom = z_sq + rho_sq + self.regularization  # (B, H, W)

        # Get constants
        deltas = self.deltas.to(device=device, dtype=dtype)
        C_delta = self.C_delta.to(device=device, dtype=dtype)
        norm_factors = self.norm_factors.to(device=device, dtype=dtype)

        # Compute propagator for each channel
        phi_channels = []
        pi_channels = []

        for c in range(self.n_channels):
            delta = deltas[c]
            C = C_delta[c]
            norm = norm_factors[c]

            # K̃_Δ = C_Δ / (e^{-2r} + ρ²)^Δ
            K_tilde = norm * C / (denom ** delta)  # (B, H, W)

            # Π̃ = 2Δ e^{-2r} C_Δ / (e^{-2r} + ρ²)^{Δ+1}
            # This is the analytic r-derivative of the UV-stabilized propagator
            pi_tilde_c = (
                norm * 2.0 * delta * z_sq * C / (denom ** (delta + 1.0))
            )  # (B, H, W)

            phi_channels.append(K_tilde)
            pi_channels.append(pi_tilde_c)

        # Stack channels: (B, C, H, W)
        phi_tilde = torch.stack(phi_channels, dim=1)
        pi_tilde = torch.stack(pi_channels, dim=1)

        return phi_tilde, pi_tilde

    def forward(self, points: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode points to UV bulk field using bulk-boundary propagator.

        Args:
            points: (B, 2) boundary point coordinates

        Returns:
            phi_tilde: (B, C, H, W) UV-stabilized field
            pi_tilde: (B, C, H, W) UV-stabilized momentum (analytic)
        """
        return self.compute_propagator(points, r=None)

    def get_grid_lengths(self) -> Tuple[float, float]:
        """Return (L_x, L_y) for FFT Laplacian eigenvalues."""
        x_min, x_max, y_min, y_max = self.domain
        return (x_max - x_min, y_max - y_min)


class FieldPointDecoder(nn.Module):
    """
    Decode fields back to point data.

    Uses center of mass (expected value) of the field as the point location.
    This is the natural inverse of HolographicPointFieldEncoder.

    Decoding: field ρ(u, v) on H×W grid → point (x, y) ∈ ℝ²

    Attributes
    ----------
    grid_size : Tuple[int, int]
        Spatial resolution
    domain : Tuple[float, ...]
        Domain bounds
    temperature : float
        Softmax temperature for soft-argmax

    Example
    -------
    >>> decoder = FieldPointDecoder(grid_size=(32, 32), domain=(-4, 4, -4, 4))
    >>> points = decoder(field)
    """

    def __init__(
        self,
        grid_size: Tuple[int, int],
        domain: Tuple[float, float, float, float] = (-4.0, 4.0, -4.0, 4.0),
        temperature: float = 1.0,
    ) -> None:
        """
        Initialize field-to-point decoder.

        Args:
            grid_size: (H, W) spatial resolution
            domain: (x_min, x_max, y_min, y_max)
            temperature: Softmax temperature for soft-argmax (lower = sharper)
        """
        super().__init__()
        self.grid_size = grid_size
        self.domain = domain
        self.temperature = temperature

        H, W = grid_size
        x_min, x_max, y_min, y_max = domain

        # Create coordinate grids
        u = torch.linspace(x_min, x_max, W)
        v = torch.linspace(y_min, y_max, H)
        V, U = torch.meshgrid(v, u, indexing="ij")

        self.register_buffer("U", U)  # (H, W)
        self.register_buffer("V", V)  # (H, W)

    def forward(self, field: Tensor, channel: int = 0) -> Tensor:
        """
        Decode field to point via center of mass.

        Args:
            field: (B, C, H, W) tensor
            channel: Which channel to use for decoding

        Returns:
            points: (B, 2) tensor of (x, y) coordinates
        """
        B = field.shape[0]
        device = field.device
        dtype = field.dtype

        # Extract single channel: (B, H, W)
        f = field[:, channel]

        U = self.U.to(device=device, dtype=dtype)  # (H, W)
        V = self.V.to(device=device, dtype=dtype)  # (H, W)

        # Soft-argmax via softmax weighting
        # This is differentiable and gives center of mass for positive fields
        f_flat = f.view(B, -1)  # (B, H*W)

        # Apply softmax with temperature
        weights = F.softmax(f_flat / self.temperature, dim=-1)  # (B, H*W)
        weights = weights.view(B, *self.grid_size)  # (B, H, W)

        # Compute expected coordinates
        x = (weights * U).sum(dim=(-2, -1))  # (B,)
        y = (weights * V).sum(dim=(-2, -1))  # (B,)

        points = torch.stack([x, y], dim=-1)  # (B, 2)
        return points


class HolographicPointFieldCodec(nn.Module):
    """
    Combined encoder-decoder for holographic point ↔ field conversion.

    This implements the EXACT AdS/CFT bulk-boundary propagator to convert
    point data to bulk field configurations.

    Unlike the Gaussian blob approach, this:
    1. Is EXACT according to AdS/CFT dictionary
    2. Automatically satisfies Klein-Gordon in the bulk
    3. Has correct UV (peaked) and IR (spread) behavior
    4. Allows the Laplacian Δ_ĝ to act nontrivially

    The encoding creates:
        Φ̃(r, x) = Σ_i K̃_Δ(r, x; x_i)

    where K̃_Δ is the UV-stabilized bulk-boundary propagator.

    Example
    -------
    >>> codec = HolographicPointFieldCodec(
    ...     d=2, deltas=[1.5, 2.0],
    ...     grid_size=(32, 32), domain=(-4, 4, -4, 4)
    ... )
    >>> phi_tilde, pi_tilde = codec.encode(points)
    >>> points_decoded = codec.decode(phi_tilde)
    """

    def __init__(
        self,
        d: int,
        deltas: Sequence[float],
        grid_size: Tuple[int, int] = (32, 32),
        domain: Tuple[float, float, float, float] = (-4.0, 4.0, -4.0, 4.0),
        r_uv: float = 1.0,
        regularization: float = 1e-6,
        temperature: float = 0.1,
        normalize_integral: bool = True,
    ) -> None:
        """
        Initialize holographic point-field codec.

        Args:
            d: Boundary dimension
            deltas: Conformal dimensions per channel
            grid_size: (H, W) spatial resolution
            domain: (x_min, x_max, y_min, y_max)
            r_uv: UV radius for the lift
            regularization: Small ε for numerical stability
            temperature: Softmax temperature for decoding
            normalize_integral: Normalize propagator to integrate to ~1
        """
        super().__init__()

        self.encoder = HolographicPointFieldEncoder(
            d=d,
            deltas=deltas,
            grid_size=grid_size,
            domain=domain,
            r_uv=r_uv,
            regularization=regularization,
            normalize_integral=normalize_integral,
        )

        self.decoder = FieldPointDecoder(
            grid_size=grid_size,
            domain=domain,
            temperature=temperature,
        )

        self.d = d
        self.n_channels = len(deltas)
        self.grid_size = grid_size
        self.domain = domain
        self.r_uv = r_uv

    def encode(self, points: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode points to bulk field using holographic propagator.

        Args:
            points: (B, 2) point coordinates

        Returns:
            phi_tilde: (B, C, H, W) UV-stabilized field
            pi_tilde: (B, C, H, W) UV-stabilized momentum
        """
        return self.encoder(points)

    def encode_phi_only(self, points: Tensor) -> Tensor:
        """Encode points to Φ̃ field only (for compatibility)."""
        phi_tilde, _ = self.encoder(points)
        return phi_tilde

    def decode(self, field: Tensor, channel: int = 0) -> Tensor:
        """Decode field to points."""
        return self.decoder(field, channel=channel)

    def forward(self, points: Tensor) -> Tensor:
        """Round-trip: encode then decode (for testing)."""
        phi_tilde, _ = self.encode(points)
        return self.decode(phi_tilde)


# =============================================================================
# Legacy Gaussian Point-Field Encoder (for comparison)
# =============================================================================


class PointFieldEncoder(nn.Module):
    """
    Encode 2D point data as fields using Gaussian blobs.

    This is the LEGACY approach - consider using HolographicPointFieldEncoder
    for AdS/CFT-faithful encoding.

    Encoding: point (x, y) ∈ ℝ² → field ρ(u, v) on H×W grid
    where ρ(u, v) = exp(-||(u,v) - (x,y)||² / (2σ²))

    Attributes
    ----------
    grid_size : Tuple[int, int]
        Spatial resolution
    domain : Tuple[float, ...]
        Domain bounds
    sigma : float
        Gaussian blob width
    n_channels : int
        Number of output channels

    Example
    -------
    >>> encoder = PointFieldEncoder(grid_size=(32, 32), sigma=0.3)
    >>> field = encoder(points)  # (B, C, H, W)
    """

    def __init__(
        self,
        grid_size: Tuple[int, int],
        domain: Tuple[float, float, float, float] = (-4.0, 4.0, -4.0, 4.0),
        sigma: float = 0.3,
        n_channels: int = 1,
    ) -> None:
        """
        Initialize Gaussian blob encoder.

        Args:
            grid_size: (H, W) spatial resolution
            domain: (x_min, x_max, y_min, y_max)
            sigma: Gaussian blob width
            n_channels: Number of output channels
        """
        super().__init__()
        self.grid_size = grid_size
        self.domain = domain
        self.sigma = sigma
        self.n_channels = n_channels

        H, W = grid_size
        x_min, x_max, y_min, y_max = domain

        u = torch.linspace(x_min, x_max, W)
        v = torch.linspace(y_min, y_max, H)
        V, U = torch.meshgrid(v, u, indexing="ij")

        self.register_buffer("U", U)
        self.register_buffer("V", V)

    def forward(self, points: Tensor) -> Tensor:
        """
        Encode points as Gaussian blobs.

        Args:
            points: (B, 2) point coordinates

        Returns:
            field: (B, C, H, W) Gaussian blob fields
        """
        B = points.shape[0]
        device = points.device
        dtype = points.dtype

        x = points[:, 0].view(B, 1, 1)
        y = points[:, 1].view(B, 1, 1)

        U = self.U.to(device=device, dtype=dtype)
        V = self.V.to(device=device, dtype=dtype)

        dist_sq = (U - x) ** 2 + (V - y) ** 2
        field = torch.exp(-dist_sq / (2 * self.sigma ** 2))

        field = field.unsqueeze(1)
        if self.n_channels > 1:
            field = field.expand(B, self.n_channels, -1, -1).contiguous()

        return field

    def get_grid_lengths(self) -> Tuple[float, float]:
        """Return (L_x, L_y) for FFT Laplacian eigenvalues."""
        x_min, x_max, y_min, y_max = self.domain
        return (x_max - x_min, y_max - y_min)


class PointFieldCodec(nn.Module):
    """
    Combined encoder-decoder for point ↔ field conversion (LEGACY Gaussian version).

    Consider using HolographicPointFieldCodec for AdS/CFT-faithful encoding.

    Example
    -------
    >>> codec = PointFieldCodec(grid_size=(32, 32), sigma=0.3)
    >>> field = codec.encode(points)
    >>> points_decoded = codec.decode(field)
    """

    def __init__(
        self,
        grid_size: Tuple[int, int] = (32, 32),
        domain: Tuple[float, float, float, float] = (-4.0, 4.0, -4.0, 4.0),
        sigma: float = 0.3,
        n_channels: int = 1,
        temperature: float = 0.1,
    ) -> None:
        """
        Initialize legacy point-field codec.

        Args:
            grid_size: (H, W) spatial resolution
            domain: (x_min, x_max, y_min, y_max)
            sigma: Gaussian blob width
            n_channels: Number of channels
            temperature: Softmax temperature for decoding
        """
        super().__init__()
        self.encoder = PointFieldEncoder(
            grid_size=grid_size,
            domain=domain,
            sigma=sigma,
            n_channels=n_channels,
        )
        self.decoder = FieldPointDecoder(
            grid_size=grid_size,
            domain=domain,
            temperature=temperature,
        )
        self.grid_size = grid_size
        self.domain = domain
        self.n_channels = n_channels

    def encode(self, points: Tensor) -> Tensor:
        """Encode points to field."""
        return self.encoder(points)

    def decode(self, field: Tensor, channel: int = 0) -> Tensor:
        """Decode field to points."""
        return self.decoder(field, channel=channel)

    def forward(self, points: Tensor) -> Tensor:
        """Round-trip: encode then decode."""
        field = self.encode(points)
        return self.decode(field)
