"""
model.py

UV-Stabilized Flow Matching Model for AdS/CFT Generative Modelling
===================================================================

Implementation of "Holographic generative flows with AdS/CFT" paper.

This model supports TWO path types:

1. HERMITE path (--path_type hermite): Paper-faithful implementation
   - AdS-affine cubic Hermite interpolation (paper Section 3.1)
   - R_θ = (0, N_θ) - residual only affects dΠ̃/dr
   - Backbone perfectly handles dΦ̃/dr = Π̃ by Hermite construction
   - Keeps path ON-SHELL (on Lagrangian submanifold)

2. LINEAR path (--path_type linear): Practical robust default
   - Simple linear interpolation: S_t = (1-t)S_0 + tS_1
   - R_θ = (N_θ^φ, N_θ^π) - full residual needed
   - Paper notes this as "robust practical default"

Both modes implement:
- UV-stabilized variables Φ̃ = e^{Δr}Φ (paper eq. 18)
- Full Klein-Gordon backbone (paper eqs. 16-17, 20-21)
- Backbone-subtracted loss (paper eq. 35)
- Slice measure weighting ||·||²_{g_{r(t)}}

=============================================================================
HOLOGRAPHIC POINT-FIELD ENCODING (Paper Section 3.3)
=============================================================================

This model supports AdS/CFT-faithful encoding of point data using the
bulk-to-boundary propagator (paper eq. 6):

    K_Δ(r, x; x') = C_Δ / (e^r |x-x'|² + e^{-r})^Δ

Enable with: use_spectral_encoding=True

This gives genuine holographic structure:
- Point sources create bulk field configurations
- The Laplacian Δ̂_g acts nontrivially on the field
- Correct UV (peaked) and IR (spread) behavior
- The momentum Π̃ is computed analytically from the propagator

=============================================================================
PAPER EQUATION REFERENCES
=============================================================================

Paper Section 2: Klein-Gordon theory in AdS/CFT
    - Metric (eq. 1): ds² = dr² + f(r)² ĝ_ab dx^a dx^b
    - Planar: f(r) = e^r, f'/f = 1
    - Klein-Gordon (eq. 3): (∂_r² + d·f'/f·∂_r + f⁻²Δ̂_g - m²)Φ = 0
    - Mass-dimension: m² = Δ(Δ-d)
    - First-order (eqs. 16-17):
        ∂_r Φ = Π
        ∂_r Π = (m² - f⁻²Δ̂_g)Φ - d(f'/f)Π

Paper Section 2.4: Boundary stabilization (UV-stable variables)
    - Definition (eq. 18): Φ̃ = e^{Δr} Φ,  Π̃ = e^{Δr}(Π + ΔΦ)
    - Stabilized ODE (eqs. 20-21, planar):
        dφ̃_k/dr = π̃_k
        dπ̃_k/dr = |k|² e^{-2r} φ̃_k - (d - 2Δ)π̃_k
    - Propagator modes (eq. 22):
        κ̃_{|k|}(r) = (2/Γ(ν)) (|k|e^r/2)^ν K_ν(|k|e^{-r})

Paper Section 3: Elements of flow matching
    - Affine parameter (eq. 29, planar):
        u(r) = [1 - e^{-d(r - r_IR)}] / [1 - e^{-d(r_UV - r_IR)}]
    - Loss (eq. 35):
        L(θ) = <||R_θ - (U_t - δ_r V_KG)||²_{g_{r(t)}}>_t

Paper Algorithm 1 (Training):
    1. Sample y ~ p_data, S_0 ~ μ_IR, set S_1 = Lift(y)
    2. Sample t ~ Uniform[0,1], form S_t and target U_t
    3. Compute backbone V_KG(S_t, r(t))
    4. Train: ||R_θ(S_t, t) - (U_t - δ_r V_KG)||²

Paper Algorithm 2 (Sampling):
    1. Sample S(r_IR) ~ μ_IR
    2. Integrate dS/dr = V_KG(S, r) + R_θ(S, r) from r_IR to r_UV
    3. Return y_gen = Readout(S(r_UV))
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ads_cft.geometry import (
    AdSGeometry,
    GeometryConfig,
    SliceGeometry,
    make_geometry,
)
from ads_cft.config import (
    PathType,
    PathConfig,
    FieldEncodingType,
    FlowModelConfig,
    DiscretizationConfig,
    IntegratorConfig,
    ODESolver,
    ResidualType,
)
from ads_cft.networks import (
    BulkState,
    CanonicalState,
    BulkField,
    AdSBackbone,
    ResidualVelocityNet,
    ResidualVelocityNet2D,
    ResidualVelocityNet2DSimple,
    PotentialResidualMLP,
    PotentialResidualCNN,
    UVDictionaryLift,
    UVReadout,
    AdSAffineHermiteCoupling,
    build_slice_laplacian,
    bulk_to_canonical,
    canonical_to_bulk,
)
from ads_cft.evolution import integrate_ode, integrate_ode_general
from ads_cft.losses import (
    compute_flow_matching_loss,
    linear_path_interpolation,
)
from ads_cft.registry import register_model

Tensor = torch.Tensor


# =============================================================================
# Note: PathType, PathConfig, FieldEncodingType, FlowModelConfig are imported from config.py
# =============================================================================


# =============================================================================
# FULL AdS Backbone (Document Section 3-4, eq uv-stable-ode)
# =============================================================================


class FullAdSBackbone(nn.Module):
    """
    Complete Klein-Gordon backbone in UV-stabilized variables.

    Document eq (uv-stable-ode):
        ∂_r Φ̃^(i) = Π̃^(i)
        ∂_r Π̃^(i) = -(d·f'/f - 2Δ_i)Π̃^(i) - (1/f²)Δ_ĝ Φ̃^(i) + dΔ_i(f'/f - 1)Φ̃^(i)

    For planar slicing (f'/f = 1):
        ∂_r Φ̃ = Π̃
        ∂_r Π̃ = -(d - 2Δ)Π̃ - (1/f²)Δ_ĝ Φ̃

    For point data (no spatial structure, Δ_ĝ = 0):
        ∂_r Φ̃ = Π̃
        ∂_r Π̃ = (2Δ - d)Π̃

    Attributes
    ----------
    geometry : AdSGeometry
        AdS geometry with warp factor
    d : int
        Boundary dimension
    n_channels : int
        Number of field channels
    slice_op : Optional[SliceLaplacian]
        Slice Laplacian operator
    spectral_encoding : bool
        Whether using spectral encoding (affects channel handling)
    """

    def __init__(
        self,
        geometry: AdSGeometry,
        d: int,
        deltas: Tuple[float, ...],
        slice_op: Optional[nn.Module] = None,
        spectral_encoding: bool = False,
    ) -> None:
        """
        Initialize full AdS backbone.

        Args:
            geometry: AdS geometry object
            d: Boundary dimension
            deltas: Conformal dimensions per channel
            slice_op: Optional slice Laplacian operator
            spectral_encoding: Whether using spectral encoding
        """
        super().__init__()
        self.geometry = geometry
        self.d = d
        self.deltas = deltas
        self.n_channels = len(deltas)
        self.slice_op = slice_op
        self.spectral_encoding = spectral_encoding

        # Pre-compute channel coefficients
        deltas_t = torch.tensor(deltas, dtype=torch.float32)

        # For spectral encoding, duplicate deltas for real/imag parts
        if spectral_encoding:
            deltas_t = deltas_t.repeat_interleave(2)

        self.register_buffer("deltas_tensor", deltas_t)
        self.register_buffer("two_delta", 2.0 * deltas_t)
        self.register_buffer("d_times_delta", float(d) * deltas_t)

    def forward(self, state: BulkState, r: Tensor) -> BulkState:
        """
        Compute backbone velocity V_KG(S, r).

        Args:
            state: BulkState with phi_tilde (B, C, *) and pi_tilde (B, C, *)
            r: Radius tensor (scalar or (B,))

        Returns:
            BulkState with dΦ̃/dr and dΠ̃/dr from backbone only
        """
        phi_tilde = state.phi_tilde
        pi_tilde = state.pi_tilde
        B = phi_tilde.shape[0]
        device = phi_tilde.device
        dtype = phi_tilde.dtype

        # Get geometry factors at radius r
        f_prime_over_f = self.geometry.log_f_prime(r)
        f_inv_sq = self.geometry.f_inv2(r)

        # Ensure tensors are on correct device
        if not isinstance(f_prime_over_f, Tensor):
            f_prime_over_f = torch.tensor(f_prime_over_f, device=device, dtype=dtype)
        else:
            f_prime_over_f = f_prime_over_f.to(device=device, dtype=dtype)
        if not isinstance(f_inv_sq, Tensor):
            f_inv_sq = torch.tensor(f_inv_sq, device=device, dtype=dtype)
        else:
            f_inv_sq = f_inv_sq.to(device=device, dtype=dtype)

        # Ensure registered buffers are on correct device
        two_delta = self.two_delta.to(device=device, dtype=dtype)
        d_times_delta = self.d_times_delta.to(device=device, dtype=dtype)

        # Broadcast geometry factors to match state shape
        ndim = phi_tilde.ndim
        if f_prime_over_f.ndim == 0:
            pass
        elif f_prime_over_f.shape[0] == B and ndim > 1:
            for _ in range(ndim - 1):
                f_prime_over_f = f_prime_over_f.unsqueeze(-1)
                f_inv_sq = f_inv_sq.unsqueeze(-1)

        # Coefficients for Π̃ equation
        d_fpf = self.d * f_prime_over_f
        two_delta_b = two_delta.view(1, -1, *([1] * (ndim - 2)))
        pi_coeff = two_delta_b - d_fpf

        # dΔ(f'/f - 1) term
        f_prime_over_f_minus_1 = f_prime_over_f - 1.0
        d_delta_b = d_times_delta.view(1, -1, *([1] * (ndim - 2)))
        phi_coeff = d_delta_b * f_prime_over_f_minus_1

        # Document eq (uv-stable-ode):
        # ∂_r Φ̃ = Π̃
        dphi_dr = pi_tilde

        # ∂_r Π̃ = (2Δ - d·f'/f)Π̃ + dΔ(f'/f - 1)Φ̃ - (1/f²)Δ_ĝΦ̃
        dpi_dr = pi_coeff * pi_tilde + phi_coeff * phi_tilde

        # Add Laplacian term if we have spatial structure
        if self.slice_op is not None:
            lap_phi = self.slice_op.apply_minus_laplacian(phi_tilde)
            dpi_dr = dpi_dr + f_inv_sq * lap_phi

        return BulkState(phi_tilde=dphi_dr, pi_tilde=dpi_dr)


# =============================================================================
# Main Model (FULL Document-Faithful Implementation)
# =============================================================================


@register_model("uv_stabilized_flow")
class UVStabilizedFlowMatchingModel(nn.Module):
    """
    FULL UV-stabilized generative flow matching model.

    Document-faithful implementation with correct path-dependent residual structure.

    Architecture:
        V_θ(S, r) = α·V_KG(S, r) + R_θ(S, r)

    where α = backbone_scale and:
        - HERMITE path: R_θ = (0, N_θ) - residual only affects dΠ̃/dr
        - LINEAR path: R_θ = (N_θ^φ, N_θ^π) - full residual

    Training (Algorithm 1):
        Loss = ||R_θ^t(S_t, t) - (U_t - V_KG^t(S_t, t))||²_{g_{r(t)}}

    Sampling (Algorithm 2):
        Integrate dS/dr = V_KG(S,r) + R_θ(S,r) from r_IR to r_UV

    ==========================================================================
    HOLOGRAPHIC ENCODING
    ==========================================================================

    When use_holographic_encoding=True, point data is encoded using the exact
    AdS/CFT bulk-boundary propagator:

        K_Δ(z, x; x') = C_Δ (z / (z² + |x - x'|²))^Δ

    This creates genuine bulk field configurations that:
    - Satisfy Klein-Gordon in the bulk
    - Have correct UV (peaked) and IR (spread) behavior
    - Allow the Laplacian to act nontrivially
    - Include analytically computed Π̃ (not just noise!)

    Attributes
    ----------
    config : FlowModelConfig
        Model configuration
    data_shape : Tuple[int, ...]
        Shape of input data
    geometry : AdSGeometry
        AdS geometry
    backbone : FullAdSBackbone
        Klein-Gordon backbone
    residual : nn.Module
        Neural residual network
    lift : UVDictionaryLift
        UV lift operation
    readout_op : UVReadout
        UV readout operation
    hermite_coupling : AdSAffineHermiteCoupling
        Hermite path coupling

    Example
    -------
    >>> config = FlowModelConfig(d=2, deltas=(1.5, 2.0), ...)
    >>> model = UVStabilizedFlowMatchingModel(config, data_shape=(2,))
    >>> samples = model.sample(batch_size=32)
    """

    def __init__(
        self,
        config: FlowModelConfig,
        data_shape: Tuple[int, ...],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        data_mean: Optional[Tensor] = None,
        data_std: Optional[Tensor] = None,
    ) -> None:
        """
        Initialize UV-stabilized flow matching model.

        Args:
            config: Model configuration
            data_shape: Shape of input data
            device: Target device
            dtype: Target dtype
            data_mean: Optional data mean for normalization
            data_std: Optional data std for normalization
        """
        super().__init__()
        self.config = config
        self.data_shape = data_shape
        self.data_dim = int(torch.tensor(data_shape).prod().item())
        self._dtype = dtype
        self._device = device or torch.device("cpu")

        # Store normalization stats
        if data_mean is not None:
            self.register_buffer("data_mean", data_mean.clone())
        else:
            self.register_buffer("data_mean", None)
        if data_std is not None:
            self.register_buffer("data_std", data_std.clone())
        else:
            self.register_buffer("data_std", None)

        # Geometry (Document Section 2)
        if config.geometry is not None:
            self.geometry = make_geometry(config.geometry)
        else:
            geom_cfg = GeometryConfig(
                d=config.d,
                slice_geometry=SliceGeometry.PLANAR,
                dtype=dtype,
                device=str(self._device),
            )
            self.geometry = make_geometry(geom_cfg)

        # Bulk field specification
        self.bulk_field = BulkField(d=config.d, deltas=config.deltas)
        self.d = config.d
        self.n_channels = len(config.deltas)

        # Radial domain
        self.r_ir = config.r_ir
        self.r_uv = config.r_uv
        self.dr = self.r_uv - self.r_ir

        self.register_buffer("_r_ir", torch.tensor(config.r_ir, dtype=dtype))
        self.register_buffer("_r_uv", torch.tensor(config.r_uv, dtype=dtype))

        # =====================================================================
        # Determine encoding type
        # =====================================================================
        if getattr(config, 'use_vanilla_cnn', False):
            self.encoding_type = FieldEncodingType.VANILLA_CNN
        elif config.use_image_encoding:
            self.encoding_type = FieldEncodingType.IMAGE
        elif config.use_spectral_encoding:
            self.encoding_type = FieldEncodingType.SPECTRAL
        elif config.use_holographic_encoding:
            self.encoding_type = FieldEncodingType.HOLOGRAPHIC_GRIDFREE
        elif config.use_holographic_grid:
            self.encoding_type = FieldEncodingType.HOLOGRAPHIC
        elif config.use_field_encoding:
            self.encoding_type = FieldEncodingType.GAUSSIAN
        else:
            self.encoding_type = FieldEncodingType.NONE

        # =====================================================================
        # Field Encoding Setup
        # =====================================================================
        self._setup_encoding(config, data_shape, dtype)

        # FULL Klein-Gordon backbone (Document Section 3-4)
        print("[DEBUG] Creating backbone...", flush=True)
        use_complex_channels = getattr(self, "_spectral_encoding", False) or getattr(
            self, "_image_encoding", False
        )
        self.backbone = FullAdSBackbone(
            geometry=self.geometry,
            d=config.d,
            deltas=config.deltas,
            slice_op=self.slice_op,
            spectral_encoding=use_complex_channels,
        )
        print("[DEBUG] Backbone created", flush=True)

        # =====================================================================
        # Residual structure depends on path type
        # =====================================================================
        self.is_hermite_path = config.path_config.path_type == PathType.ADS_AFFINE_HERMITE
        self.use_full_residual = not self.is_hermite_path

        # Create residual network
        self._setup_residual_network(config)

        # UV lift (Document Section 5.1)
        print("[DEBUG] Creating UV lift and readout...", flush=True)
        self.lift = UVDictionaryLift(
            geometry=self.geometry,
            bulk_field=self.bulk_field,
            disc=self.disc,
            embed_src=None,
            embed_vev=self._identity_embed,
            noise_sigma=config.lift_noise_sigma,
        )

        # UV readout (Document Section 5.2)
        self.readout_op = UVReadout(
            d=config.d,
            deltas=config.deltas,
            mode="direct",
        )

        # AdS-affine Hermite coupling (Document Section 9)
        self.hermite_coupling = AdSAffineHermiteCoupling(
            geometry=self.geometry,
            r_ir=self.r_ir,
            r_uv=self.r_uv,
            use_radial_hermite=config.use_radial_hermite,
        )

        # Config references
        self.integrator_config = config.integrator
        self.path_config = config.path_config

        # Log configuration
        self._log_configuration(config)
        print("[DEBUG] UVStabilizedFlowMatchingModel.__init__ complete!", flush=True)

    def _setup_encoding(
        self, config: FlowModelConfig, data_shape: Tuple[int, ...], dtype: torch.dtype
    ) -> None:
        """Set up field encoding based on encoding type."""
        if self.encoding_type == FieldEncodingType.VANILLA_CNN:
            self._setup_vanilla_cnn_encoding(config, data_shape, dtype)
        elif self.encoding_type == FieldEncodingType.IMAGE:
            self._setup_image_encoding(config, data_shape, dtype)
        elif self.encoding_type == FieldEncodingType.SPECTRAL:
            self._setup_spectral_encoding(config, dtype)
        elif self.encoding_type == FieldEncodingType.HOLOGRAPHIC_GRIDFREE:
            self._setup_holographic_gridfree_encoding(config)
        elif self.encoding_type == FieldEncodingType.HOLOGRAPHIC:
            self._setup_holographic_grid_encoding(config, dtype)
        elif self.encoding_type == FieldEncodingType.GAUSSIAN:
            self._setup_gaussian_encoding(config, dtype)
        else:
            self._setup_no_encoding(config, dtype)

    def _setup_image_encoding(
        self, config: FlowModelConfig, data_shape: Tuple[int, ...], dtype: torch.dtype
    ) -> None:
        """Set up image FFT encoding."""
        from ads_cft.encoding_image import ImageSpectralCodec

        print("[DEBUG] Starting IMAGE encoding setup...", flush=True)

        # Determine image shape
        if config.image_shape is not None:
            image_shape = config.image_shape
        elif len(data_shape) == 3:
            image_shape = data_shape
        elif len(data_shape) == 2:
            image_shape = (1,) + data_shape
        else:
            raise ValueError(f"Cannot determine image shape from data_shape={data_shape}")

        C, H, W = image_shape
        print(f"[IMAGE] Using direct FFT encoding for images", flush=True)
        print(f"[IMAGE] Image shape: {C}×{H}×{W}", flush=True)

        self.d = 2
        self.n_channels = C

        image_delta = config.deltas[0] if config.deltas else 2.0
        self.image_codec = ImageSpectralCodec(image_shape, delta=image_delta, r_uv=config.r_uv)

        # Laplacian
        self.slice_op = self.image_codec.get_laplacian()

        disc_image = DiscretizationConfig(
            representation="spectral",
            grid_shape=(H, W),
            box_lengths=(float(H), float(W)),
        )
        self.disc = disc_image

        self.is_point_data = False
        self.field_shape = self.image_codec.field_shape
        self.image_shape = image_shape
        self.point_field_codec = None
        self.holographic_codec = None
        self.spectral_codec = None
        self.use_field_encoding = True
        self.use_holographic_encoding = False
        self._holographic_gridfree = False
        self._spectral_encoding = False
        self._image_encoding = True

    def _setup_vanilla_cnn_encoding(
        self, config: FlowModelConfig, data_shape: Tuple[int, ...], dtype: torch.dtype
    ) -> None:
        """
        Set up vanilla CNN encoding for flow matching baseline.
        
        This is a non-physics baseline that:
        - Takes raw images directly (no FFT, no bulk-boundary propagator)
        - Uses CNN architecture on pixel space
        - Does vanilla linear flow matching
        
        The "phi_tilde" channel holds the raw image.
        The "pi_tilde" channel holds zeros (or noise for regularization).
        """
        print("[DEBUG] Starting VANILLA_CNN encoding setup...", flush=True)

        # Determine image shape
        if config.image_shape is not None:
            image_shape = config.image_shape
        elif len(data_shape) == 3:
            image_shape = data_shape
        elif len(data_shape) == 2:
            image_shape = (1,) + data_shape
        else:
            raise ValueError(f"Cannot determine image shape from data_shape={data_shape}")

        C, H, W = image_shape
        print(f"[VANILLA_CNN] Using raw image encoding (no physics)", flush=True)
        print(f"[VANILLA_CNN] Image shape: {C}×{H}×{W}", flush=True)

        self.d = 2
        self.n_channels = C
        self._vanilla_image_shape = (C, H, W)
        self._vanilla_noise_sigma = config.lift_noise_sigma

        # Discretization config (for compatibility)
        self.disc = DiscretizationConfig(
            representation="physical",
            grid_shape=(H, W),
            box_lengths=(1.0, 1.0),
        )

        # Slice operator is None for vanilla (no Laplacian needed)
        self.slice_op = None

        # Field shape: 2*C channels for (phi, pi) structure
        # Using 2*C to maintain compatibility with CNN architecture
        self.field_shape = (2 * C, H, W)
        self.is_point_data = False
        self.point_field_codec = None
        self.holographic_codec = None
        self.spectral_codec = None
        self.image_codec = None
        self.use_field_encoding = True
        self.use_holographic_encoding = False
        self._holographic_gridfree = False
        self._spectral_encoding = False
        self._image_encoding = False
        self._vanilla_cnn = True

    def _setup_spectral_encoding(self, config: FlowModelConfig, dtype: torch.dtype) -> None:
        """Set up spectral holographic encoding."""
        from ads_cft.encoding_spectral import SpectralHolographicCodec

        K = config.spectral_n_modes
        d = config.d

        print(f"[SPECTRAL] Using spectral bulk-boundary propagator (d={d})", flush=True)
        print(f"[SPECTRAL] Modes: {K}×{K} = {K*K} coefficients", flush=True)

        # Get hsv_p if this is an HSV geometry (for correct spectral propagator selection)
        hsv_p = getattr(self.geometry, 'p', None)
        
        # Get hsv_propagator_mode from config
        hsv_propagator_mode = getattr(config, 'hsv_propagator_mode', 'bessel')

        self.spectral_codec = SpectralHolographicCodec(
            d=config.d,
            deltas=config.deltas,
            n_modes=K,
            r_uv=config.r_uv,
            domain=config.field_domain,
            regularization=config.holographic_regularization,
            decode_resolution=config.spectral_decode_resolution,
            geometry=self.geometry.slice_geometry,
            hsv_p=hsv_p,
            hsv_propagator_mode=hsv_propagator_mode,
        )

        self.slice_op = self.spectral_codec.get_laplacian()

        disc_spectral = DiscretizationConfig(
            representation="physical",
            grid_shape=(K, K),
            box_lengths=(
                config.field_domain[1] - config.field_domain[0],
                config.field_domain[3] - config.field_domain[2],
            ),
        )
        self.disc = disc_spectral

        self.field_shape = (2 * self.n_channels, K, K)
        self.is_point_data = False
        self.point_field_codec = None
        self.holographic_codec = None
        self.use_field_encoding = True
        self.use_holographic_encoding = False
        self._holographic_gridfree = False
        self._spectral_encoding = True
        self._spectral_decode_method = config.spectral_decode_method
        self._use_general_spectral = False

    def _setup_holographic_gridfree_encoding(self, config: FlowModelConfig) -> None:
        """Set up grid-free holographic encoding."""
        from ads_cft.encoding_spectral import HolographicPointCodec

        print(f"[HOLOGRAPHIC-GRIDFREE] Using grid-free bulk-boundary propagator", flush=True)

        self.holographic_codec = HolographicPointCodec(
            d=config.d,
            deltas=config.deltas,
            r_uv=config.r_uv,
            regularization=config.holographic_regularization,
        )

        self.disc = config.discretization
        self.is_point_data = True
        self.slice_op = None
        self.field_shape = None
        self.point_field_codec = None
        self.spectral_codec = None
        self.use_field_encoding = False
        self.use_holographic_encoding = True
        self._holographic_gridfree = True
        self._spectral_encoding = False

    def _setup_holographic_grid_encoding(
        self, config: FlowModelConfig, dtype: torch.dtype
    ) -> None:
        """Set up grid-based holographic encoding."""
        from ads_cft.encoding_spectral import HolographicPointFieldCodec

        print(f"[HOLOGRAPHIC-GRID] Using grid-based bulk-boundary propagator", flush=True)

        self.holographic_codec = HolographicPointFieldCodec(
            d=config.d,
            deltas=config.deltas,
            grid_size=config.field_grid_size,
            domain=config.field_domain,
            r_uv=config.r_uv,
            regularization=config.holographic_regularization,
            temperature=config.field_temperature,
            normalize_integral=config.holographic_normalize,
        )

        x_min, x_max, y_min, y_max = config.field_domain
        box_lengths = (x_max - x_min, y_max - y_min)

        disc_with_grid = DiscretizationConfig(
            representation="physical",
            grid_shape=config.field_grid_size,
            box_lengths=box_lengths,
        )
        self.disc = disc_with_grid

        self.is_point_data = False
        self.slice_op = build_slice_laplacian(
            self.geometry, disc_with_grid, device=self._device, dtype=dtype
        )

        self.field_shape = (self.n_channels, *config.field_grid_size)
        self.point_field_codec = None
        self.spectral_codec = None
        self.use_field_encoding = True
        self.use_holographic_encoding = True
        self._holographic_gridfree = False
        self._spectral_encoding = False

    def _setup_gaussian_encoding(self, config: FlowModelConfig, dtype: torch.dtype) -> None:
        """Set up legacy Gaussian blob encoding."""
        from ads_cft.encoding_spectral import PointFieldCodec

        print(f"[GAUSSIAN] Using legacy Gaussian blob encoding", flush=True)

        self.point_field_codec = PointFieldCodec(
            grid_size=config.field_grid_size,
            domain=config.field_domain,
            sigma=config.field_sigma,
            n_channels=self.n_channels,
            temperature=config.field_temperature,
        )

        x_min, x_max, y_min, y_max = config.field_domain
        box_lengths = (x_max - x_min, y_max - y_min)

        disc_with_grid = DiscretizationConfig(
            representation="physical",
            grid_shape=config.field_grid_size,
            box_lengths=box_lengths,
        )
        self.disc = disc_with_grid

        self.is_point_data = False
        self.slice_op = build_slice_laplacian(
            self.geometry, disc_with_grid, device=self._device, dtype=dtype
        )

        self.field_shape = (self.n_channels, *config.field_grid_size)
        self.holographic_codec = None
        self.spectral_codec = None
        self.use_field_encoding = True
        self.use_holographic_encoding = False
        self._holographic_gridfree = False
        self._spectral_encoding = False

    def _setup_no_encoding(self, config: FlowModelConfig, dtype: torch.dtype) -> None:
        """Set up for point data or native field data."""
        self.point_field_codec = None
        self.holographic_codec = None
        self.spectral_codec = None
        self.is_point_data = config.discretization.grid_shape is None
        self.disc = config.discretization

        if self.is_point_data:
            self.slice_op = None
        else:
            self.slice_op = build_slice_laplacian(
                self.geometry, config.discretization, device=self._device, dtype=dtype
            )
        self.field_shape = None
        self.use_field_encoding = False
        self.use_holographic_encoding = False
        self._holographic_gridfree = False
        self._spectral_encoding = False

    def _setup_residual_network(self, config: FlowModelConfig) -> None:
        """Set up residual velocity network based on residual_type."""
        
        # Store residual type
        self.residual_type = config.residual_type
        
        use_cnn = (
            self.encoding_type
            in [
                FieldEncodingType.HOLOGRAPHIC,
                FieldEncodingType.GAUSSIAN,
                FieldEncodingType.SPECTRAL,
                FieldEncodingType.IMAGE,
                FieldEncodingType.VANILLA_CNN,
            ]
            or (not self.is_point_data and self.disc.grid_shape is not None)
        )

        print(f"[DEBUG] Creating residual network (use_cnn={use_cnn}, type={config.residual_type.value})...", flush=True)
        
        if config.residual_type == ResidualType.POTENTIAL:
            # Potential-based residual for symplectic integration
            # The network outputs U_θ(Φ, r), force is -ω ∂U_θ/∂Φ
            if use_cnn:
                if self.encoding_type in [FieldEncodingType.SPECTRAL, FieldEncodingType.IMAGE]:
                    # For spectral/image: phi_tilde has 2*n_channels (real+imag for complex coefficients)
                    in_ch = 2 * self.n_channels  # Only Φ, not Π (potential depends on position only)
                elif self.encoding_type == FieldEncodingType.VANILLA_CNN:
                    in_ch = self.n_channels
                else:
                    in_ch = self.n_channels
                    
                cnn_hidden = config.residual_hidden if config.residual_hidden is not None else config.cnn_hidden
                cnn_depth = config.residual_depth if config.residual_depth is not None else config.cnn_depth
                
                # Auto-limit depth
                if self.disc.grid_shape is not None:
                    min_spatial = min(self.disc.grid_shape)
                    import math
                    max_safe_depth = max(1, int(math.log2(min_spatial)))
                    if cnn_depth > max_safe_depth:
                        print(f"[WARNING] Auto-limiting CNN depth to {max_safe_depth}", flush=True)
                        cnn_depth = max_safe_depth
                
                print(f"[DEBUG] Potential CNN: in_ch={in_ch}, hidden={cnn_hidden}, depth={cnn_depth}", flush=True)
                self.residual = PotentialResidualCNN(
                    in_channels=in_ch,
                    hidden_channels=cnn_hidden,
                    depth=cnn_depth,
                    time_emb_dim=config.residual_time_emb_dim,
                )
            else:
                mlp_hidden = config.residual_hidden if config.residual_hidden is not None else config.mlp_hidden
                mlp_depth = config.residual_depth if config.residual_depth is not None else config.mlp_depth
                
                print(f"[DEBUG] Potential MLP: channels={self.n_channels}, hidden={mlp_hidden}, depth={mlp_depth}", flush=True)
                self.residual = PotentialResidualMLP(
                    channels=self.n_channels,
                    hidden=mlp_hidden,
                    depth=mlp_depth,
                    time_emb_dim=config.residual_time_emb_dim,
                )
        else:
            # DIRECT residual (default): R_θ = (0, N_θ) or (N_θ^φ, N_θ^π)
            if use_cnn:
                if self.encoding_type == FieldEncodingType.SPECTRAL:
                    # Use simplified architecture for spectral encoding (toy datasets)
                    # This matches the original 10M parameter architecture
                    in_ch = 2 * self.n_channels
                    print(f"[DEBUG] Using simplified 2D CNN for spectral data ({in_ch} input channels)", flush=True)
                    
                    cnn_hidden = config.residual_hidden if config.residual_hidden is not None else config.cnn_hidden
                    cnn_depth = config.residual_depth if config.residual_depth is not None else config.cnn_depth
                    
                    # Auto-limit depth based on spatial size to prevent crash
                    if self.disc.grid_shape is not None:
                        min_spatial = min(self.disc.grid_shape)
                        import math
                        max_safe_depth = max(1, int(math.log2(min_spatial)))
                        if cnn_depth > max_safe_depth:
                            print(f"[WARNING] residual_depth={cnn_depth} too deep for {self.disc.grid_shape} spatial size", flush=True)
                            print(f"[WARNING] Auto-limiting CNN depth to {max_safe_depth}", flush=True)
                            cnn_depth = max_safe_depth
                    
                    print(f"[DEBUG] CNN params: in_ch={in_ch}, hidden={cnn_hidden}, depth={cnn_depth}, time_emb={config.residual_time_emb_dim}, output_both={self.use_full_residual}", flush=True)
                    
                    self.residual = ResidualVelocityNet2DSimple(
                        in_channels=in_ch,
                        hidden=cnn_hidden,
                        depth=cnn_depth,
                        time_emb_dim=config.residual_time_emb_dim,
                        output_both=self.use_full_residual,
                    )
                    print(f"[DEBUG] ResidualVelocityNet2DSimple created successfully", flush=True)
                    
                elif self.encoding_type == FieldEncodingType.IMAGE:
                    # Use full architecture for image encoding (MNIST, etc.)
                    in_ch = 2 * self.n_channels
                    print(f"[DEBUG] Using 2D CNN for image data ({in_ch} input channels)", flush=True)
                    
                    cnn_hidden = config.residual_hidden if config.residual_hidden is not None else config.cnn_hidden
                    cnn_depth = config.residual_depth if config.residual_depth is not None else config.cnn_depth
                    
                    # Auto-limit depth based on spatial size to prevent crash
                    if self.disc.grid_shape is not None:
                        min_spatial = min(self.disc.grid_shape)
                        import math
                        max_safe_depth = max(1, int(math.log2(min_spatial)))
                        if cnn_depth > max_safe_depth:
                            print(f"[WARNING] residual_depth={cnn_depth} too deep for {self.disc.grid_shape} spatial size", flush=True)
                            print(f"[WARNING] Auto-limiting CNN depth to {max_safe_depth}", flush=True)
                            cnn_depth = max_safe_depth
                    
                    print(f"[DEBUG] CNN params: in_ch={in_ch}, hidden={cnn_hidden}, depth={cnn_depth}, time_emb={config.residual_time_emb_dim}, output_both={self.use_full_residual}", flush=True)
                    
                    self.residual = ResidualVelocityNet2D(
                        in_channels=in_ch,
                        hidden=cnn_hidden,
                        depth=cnn_depth,
                        time_emb_dim=config.residual_time_emb_dim,
                        output_both=self.use_full_residual,
                    )
                    print(f"[DEBUG] ResidualVelocityNet2D created successfully", flush=True)
                    
                elif self.encoding_type == FieldEncodingType.VANILLA_CNN:
                    in_ch = self.n_channels
                    print(f"[DEBUG] Using 2D CNN for vanilla flow matching ({in_ch} input channels)", flush=True)
                    
                    cnn_hidden = config.residual_hidden if config.residual_hidden is not None else config.cnn_hidden
                    cnn_depth = config.residual_depth if config.residual_depth is not None else config.cnn_depth
                    
                    # Auto-limit depth based on spatial size to prevent crash
                    if self.disc.grid_shape is not None:
                        min_spatial = min(self.disc.grid_shape)
                        import math
                        max_safe_depth = max(1, int(math.log2(min_spatial)))
                        if cnn_depth > max_safe_depth:
                            print(f"[WARNING] residual_depth={cnn_depth} too deep for {self.disc.grid_shape} spatial size", flush=True)
                            print(f"[WARNING] Auto-limiting CNN depth to {max_safe_depth}", flush=True)
                            cnn_depth = max_safe_depth
                    
                    print(f"[DEBUG] CNN params: in_ch={in_ch}, hidden={cnn_hidden}, depth={cnn_depth}, time_emb={config.residual_time_emb_dim}, output_both={self.use_full_residual}", flush=True)
                    
                    self.residual = ResidualVelocityNet2D(
                        in_channels=in_ch,
                        hidden=cnn_hidden,
                        depth=cnn_depth,
                        time_emb_dim=config.residual_time_emb_dim,
                        output_both=self.use_full_residual,
                    )
                    print(f"[DEBUG] ResidualVelocityNet2D created successfully", flush=True)
                else:
                    in_ch = self.n_channels
                    print(f"[DEBUG] Using 2D CNN for field data ({in_ch} input channels)", flush=True)

                    cnn_hidden = config.residual_hidden if config.residual_hidden is not None else config.cnn_hidden
                    cnn_depth = config.residual_depth if config.residual_depth is not None else config.cnn_depth
                    
                    # Auto-limit depth based on spatial size to prevent crash
                    if self.disc.grid_shape is not None:
                        min_spatial = min(self.disc.grid_shape)
                        import math
                        max_safe_depth = max(1, int(math.log2(min_spatial)))
                        if cnn_depth > max_safe_depth:
                            print(f"[WARNING] residual_depth={cnn_depth} too deep for {self.disc.grid_shape} spatial size", flush=True)
                            print(f"[WARNING] Auto-limiting CNN depth to {max_safe_depth}", flush=True)
                            cnn_depth = max_safe_depth

                    print(f"[DEBUG] CNN params: in_ch={in_ch}, hidden={cnn_hidden}, depth={cnn_depth}, time_emb={config.residual_time_emb_dim}, output_both={self.use_full_residual}", flush=True)
                    
                    self.residual = ResidualVelocityNet2D(
                        in_channels=in_ch,
                        hidden=cnn_hidden,
                        depth=cnn_depth,
                        time_emb_dim=config.residual_time_emb_dim,
                        output_both=self.use_full_residual,
                    )
                    print(f"[DEBUG] ResidualVelocityNet2D created successfully", flush=True)
            else:
                mlp_hidden = config.residual_hidden if config.residual_hidden is not None else config.mlp_hidden
                mlp_depth = config.residual_depth if config.residual_depth is not None else config.mlp_depth
                
                print(f"[DEBUG] Using MLP residual network for point data (hidden={mlp_hidden}, depth={mlp_depth})", flush=True)
                self.residual = ResidualVelocityNet(
                    channels=self.n_channels,
                    hidden=mlp_hidden,
                    depth=mlp_depth,
                    time_emb_dim=config.residual_time_emb_dim,
                    output_both=self.use_full_residual,
                )
        print("[DEBUG] Residual network created", flush=True)

    def _log_configuration(self, config: FlowModelConfig) -> None:
        """Log model configuration."""
        print(f"[PATH] Using {config.path_config.path_type.value.upper()} path", flush=True)
        if self.is_hermite_path:
            print(
                f"[HERMITE] R_θ = (0, N_θ) - residual only affects dΠ̃/dr",
                flush=True,
            )
            print(f"[HERMITE] backbone_scale = {config.backbone_scale}", flush=True)
        else:
            print(f"[LINEAR] R_θ = (N_θ^φ, N_θ^π) - full residual", flush=True)
            print(f"[LINEAR] backbone_scale = {config.backbone_scale}", flush=True)

    def _identity_embed(self, y: Tensor) -> Tensor:
        """
        Identity embedding E(y) = y.

        Document Section 5.1 "No learnable lift (minimal)": E(y) = y

        Args:
            y: Input data

        Returns:
            Unchanged input
        """
        B = y.shape[0]
        if y.ndim == 2:
            if y.shape[1] != self.n_channels:
                raise ValueError(
                    f"Data dimension {y.shape[1]} doesn't match n_channels {self.n_channels}."
                )
            return y
        return y.view(B, self.n_channels, *y.shape[2:])

    # =========================================================================
    # IR Prior (Document Section 7)
    # =========================================================================

    def sample_ir_prior(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> BulkState:
        """
        Sample from IR prior μ_IR.

        Document Section 7 (Sobolev-type Gaussian):
            φ̃_j^(i)(r_IR) ~ N(0, σ²_φ,i(λ_j))
            π̃_j^(i)(r_IR) ~ N(0, σ²_π,i(λ_j))

        where σ²(λ) = c(1+λ)^{-s}

        Args:
            batch_size: Number of samples
            generator: Optional random generator

        Returns:
            BulkState sampled from IR prior
        """
        device = self._r_ir.device
        dtype = self._r_ir.dtype
        cfg = self.config

        # Determine number of channels
        if hasattr(self, "_spectral_encoding") and self._spectral_encoding:
            n_ch = 2 * self.n_channels
        elif hasattr(self, "_image_encoding") and self._image_encoding:
            n_ch = 2 * self.n_channels
        else:
            n_ch = self.n_channels

        # Determine shape
        if self.disc.grid_shape is not None:
            spatial_shape = tuple(self.disc.grid_shape)
            shape = (batch_size, n_ch, *spatial_shape)
        else:
            shape = (batch_size, n_ch)

        # Sample with Sobolev weighting if configured
        if cfg.prior_s_phi != 0.0 or cfg.prior_s_pi != 0.0:
            if self.slice_op is not None and hasattr(self.slice_op, "diag_eigs"):
                eigs = self.slice_op.diag_eigs(device=device, dtype=dtype)
                if eigs is not None:
                    eigs_b = eigs.view(*([1, 1] + list(eigs.shape)))
                    var_phi = cfg.prior_c_phi * (1.0 + eigs_b) ** (-cfg.prior_s_phi)
                    var_pi = cfg.prior_c_pi * (1.0 + eigs_b) ** (-cfg.prior_s_pi)

                    phi_tilde = torch.sqrt(var_phi) * torch.randn(
                        shape, device=device, dtype=dtype, generator=generator
                    )
                    pi_tilde = torch.sqrt(var_pi) * torch.randn(
                        shape, device=device, dtype=dtype, generator=generator
                    )
                    return BulkState(phi_tilde=phi_tilde, pi_tilde=pi_tilde)

        # Fallback: uniform variance
        sigma_phi = math.sqrt(cfg.prior_c_phi)
        sigma_pi = math.sqrt(cfg.prior_c_pi)

        phi_tilde = sigma_phi * torch.randn(
            shape, device=device, dtype=dtype, generator=generator
        )
        pi_tilde = sigma_pi * torch.randn(
            shape, device=device, dtype=dtype, generator=generator
        )

        return BulkState(phi_tilde=phi_tilde, pi_tilde=pi_tilde)

    # =========================================================================
    # Lift (Document Section 5.1)
    # =========================================================================

    def lift_data(
        self,
        y: Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> BulkState:
        """
        Lift data to UV bulk state.

        Supports multiple encoding types:
        - IMAGE: Direct FFT with physical momentum
        - SPECTRAL: Spectral bulk-boundary propagator
        - HOLOGRAPHIC_GRIDFREE: Grid-free propagator
        - HOLOGRAPHIC: Grid-based propagator
        - GAUSSIAN: Legacy Gaussian blob
        - NONE: Direct lift

        Args:
            y: Input data
            generator: Optional random generator

        Returns:
            BulkState at UV boundary
        """
        if self.encoding_type == FieldEncodingType.VANILLA_CNN:
            # Vanilla CNN: raw images, no physics
            # y is (B, C, H, W) or (B, H, W)
            if y.ndim == 3:
                y = y.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
            
            phi = y  # Raw image as phi
            pi = torch.zeros_like(y)  # Zero momentum (vanilla flow matching)
            
            # Add small noise for regularization if needed
            noise_sigma = getattr(self, '_vanilla_noise_sigma', 0.0)
            if noise_sigma > 0:
                pi = noise_sigma * torch.randn_like(pi)
            
            return BulkState(phi_tilde=phi, pi_tilde=pi)

        elif self.encoding_type == FieldEncodingType.IMAGE:
            phi_hat, pi_hat = self.image_codec.encode_with_pi(y)
            noise_sigma = self.config.lift_noise_sigma
            pi_noise = noise_sigma * torch.randn_like(pi_hat)
            pi_hat = pi_hat + pi_noise
            return BulkState(phi_tilde=phi_hat, pi_tilde=pi_hat)

        elif self.encoding_type == FieldEncodingType.SPECTRAL:
            phi_hat, pi_hat = self.spectral_codec.encode(y)
            noise_sigma = self.config.lift_noise_sigma
            pi_noise = noise_sigma * torch.randn_like(pi_hat)
            pi_hat = pi_hat + pi_noise
            return BulkState(phi_tilde=phi_hat, pi_tilde=pi_hat)

        elif self.encoding_type == FieldEncodingType.HOLOGRAPHIC_GRIDFREE:
            phi_tilde_uv, pi_tilde_analytic, laplacian_coeff = self.holographic_codec.encode(y)
            self._last_laplacian_coeff = laplacian_coeff
            noise_sigma = self.config.lift_noise_sigma
            pi_noise = noise_sigma * torch.randn_like(pi_tilde_analytic)
            pi_tilde_uv = pi_tilde_analytic + pi_noise
            return BulkState(phi_tilde=phi_tilde_uv, pi_tilde=pi_tilde_uv)

        elif self.encoding_type == FieldEncodingType.HOLOGRAPHIC:
            phi_tilde_uv, pi_tilde_analytic = self.holographic_codec.encode(y)
            noise_sigma = self.config.lift_noise_sigma
            pi_noise = noise_sigma * torch.randn_like(pi_tilde_analytic)
            pi_tilde_uv = pi_tilde_analytic + pi_noise
            return BulkState(phi_tilde=phi_tilde_uv, pi_tilde=pi_tilde_uv)

        elif self.encoding_type == FieldEncodingType.GAUSSIAN:
            y = self.point_field_codec.encode(y)
            r_uv = self._r_uv
            state = self.lift(y, r_uv, generator=generator)
            pi_scale = self.config.hermite_pi_scale
            if pi_scale != 1.0:
                state = BulkState(
                    phi_tilde=state.phi_tilde,
                    pi_tilde=state.pi_tilde * pi_scale,
                )
            return state

        else:
            r_uv = self._r_uv
            state = self.lift(y, r_uv, generator=generator)
            pi_scale = self.config.hermite_pi_scale
            if pi_scale != 1.0:
                state = BulkState(
                    phi_tilde=state.phi_tilde,
                    pi_tilde=state.pi_tilde * pi_scale,
                )
            return state

    # =========================================================================
    # Readout (Document Section 5.2)
    # =========================================================================

    def readout(self, state: BulkState) -> Tensor:
        """
        Read out data from UV bulk state.

        Document Section 5.2 "Stabilized direct readout":
            y_gen = Readout(Φ̃(r_UV))

        Args:
            state: UV bulk state

        Returns:
            Generated data
        """
        if self.encoding_type == FieldEncodingType.VANILLA_CNN:
            # Vanilla CNN: return phi directly, clipped to [0, 1]
            return state.phi_tilde.clamp(0, 1)
        elif self.encoding_type == FieldEncodingType.IMAGE:
            return self.image_codec.decode(state.phi_tilde)
        elif self.encoding_type == FieldEncodingType.SPECTRAL:
            return self.spectral_codec.decode(
                state.phi_tilde, method=self._spectral_decode_method
            )
        elif self.encoding_type == FieldEncodingType.HOLOGRAPHIC_GRIDFREE:
            return self.holographic_codec.decode(state.phi_tilde)
        elif self.encoding_type == FieldEncodingType.HOLOGRAPHIC:
            return self.holographic_codec.decode(state.phi_tilde)
        elif self.encoding_type == FieldEncodingType.GAUSSIAN:
            return self.point_field_codec.decode(state.phi_tilde)

        return state.phi_tilde.view(-1, *self.data_shape)

    # =========================================================================
    # FULL Velocity Field (Document Section 4.2)
    # =========================================================================

    def velocity(self, state: BulkState, r: Tensor) -> BulkState:
        """
        FULL velocity field V_θ(S, r) = α·V_KG(S, r) + R_θ(S, r).

        Document Section 4.2 - PATH-DEPENDENT RESIDUAL STRUCTURE:
        - HERMITE path: R_θ = (0, N_θ) - backbone handles dΦ̃/dr = Π̃
        - LINEAR path: R_θ = (N_θ^φ, N_θ^π) - full residual needed

        For POTENTIAL residual type:
        - R_θ = (0, -ω ∂U_θ/∂Φ) - derived from learned potential

        Args:
            state: Current bulk state
            r: Current radius

        Returns:
            Velocity (dΦ̃/dr, dΠ̃/dr)
        """
        t = (r - self._r_ir) / (self._r_uv - self._r_ir)
        t = t.clamp(0, 1)

        alpha = self.config.backbone_scale
        
        # Skip backbone computation entirely when not needed (vanilla CNN or backbone_scale=0)
        if alpha == 0.0:
            # Pure neural residual - no physics
            if self.is_hermite_path:
                n_pi = self.residual(state.phi_tilde, state.pi_tilde, t)
                return BulkState(
                    phi_tilde=torch.zeros_like(state.phi_tilde),
                    pi_tilde=n_pi,
                )
            else:
                n_phi, n_pi = self.residual(state.phi_tilde, state.pi_tilde, t)
                return BulkState(phi_tilde=n_phi, pi_tilde=n_pi)

        # Compute backbone only when needed
        V_KG = self.backbone(state, r)

        if self.residual_type == ResidualType.POTENTIAL:
            omega = self.geometry.slice_weight(r)
            force = self.residual.compute_force(state.phi_tilde, t, omega)
            return BulkState(
                phi_tilde=alpha * V_KG.phi_tilde,
                pi_tilde=alpha * V_KG.pi_tilde + force,
            )
        elif self.is_hermite_path:
            n_pi = self.residual(state.phi_tilde, state.pi_tilde, t)
            return BulkState(
                phi_tilde=alpha * V_KG.phi_tilde,
                pi_tilde=alpha * V_KG.pi_tilde + n_pi,
            )
        else:
            n_phi, n_pi = self.residual(state.phi_tilde, state.pi_tilde, t)
            return BulkState(
                phi_tilde=alpha * V_KG.phi_tilde + n_phi,
                pi_tilde=alpha * V_KG.pi_tilde + n_pi,
            )

    def _symplectic_force(self, phi: Tensor, P: Tensor, r: Tensor) -> Tensor:
        """
        Compute the total force for symplectic integration in canonical variables.
        
        Instead of computing F_KG from scratch, we use the backbone + residual
        (which is what the residual was trained against) and transform to canonical.
        
        Args:
            phi: Physical field Φ (not UV-stabilized)
            P: Canonical momentum P = ωΠ
            r: Radial coordinate
            
        Returns:
            Force dP/dr
        """
        device = phi.device
        dtype = phi.dtype
        
        if isinstance(r, (int, float)):
            r = torch.tensor(r, device=device, dtype=dtype)
        
        # Get geometry quantities
        f = self.geometry.f(r)
        if not isinstance(f, Tensor):
            f = torch.tensor(f, device=device, dtype=dtype)
        f_prime_over_f = self.geometry.log_f_prime(r)
        if not isinstance(f_prime_over_f, Tensor):
            f_prime_over_f = torch.tensor(f_prime_over_f, device=device, dtype=dtype)
        omega = self.geometry.slice_weight(r)
        if not isinstance(omega, Tensor):
            omega = torch.tensor(omega, device=device, dtype=dtype)
        
        # Expand for broadcasting
        if f.ndim < phi.ndim:
            f = f.view(*([1] * (phi.ndim - f.ndim)), *f.shape).expand_as(phi)
        if f_prime_over_f.ndim < phi.ndim:
            f_prime_over_f = f_prime_over_f.view(*([1] * (phi.ndim - f_prime_over_f.ndim)), *f_prime_over_f.shape).expand_as(phi)
        if omega.ndim < phi.ndim:
            omega = omega.view(*([1] * (phi.ndim - omega.ndim)), *omega.shape).expand_as(phi)
        
        # Get delta for stabilization
        delta = self.bulk_field.deltas[0]
        
        # Reconstruct UV-stabilized state from canonical (Φ, P)
        # Φ̃ = f^Δ · Φ
        # Π = P / ω
        # Π̃ = f^Δ · (Π + Δ·(f'/f)·Φ)
        f_delta = f ** delta
        pi = P / omega.clamp(min=1e-10)
        
        phi_tilde = f_delta * phi
        pi_tilde = f_delta * (pi + delta * f_prime_over_f * phi)
        
        state = BulkState(phi_tilde=phi_tilde, pi_tilde=pi_tilde)
        
        # Get full velocity in UV-stabilized space (backbone + residual)
        V = self.velocity(state, r)
        
        # V.pi_tilde is dΠ̃/dr. We need dP/dr.
        # 
        # From P = ω·Π and Π̃ = f^Δ·(Π + Δ·(f'/f)·Φ):
        #   Π = f^{-Δ}·Π̃ - Δ·(f'/f)·Φ
        #   P = ω·Π = ω·f^{-Δ}·Π̃ - ω·Δ·(f'/f)·Φ
        #
        # Taking d/dr and using dΦ/dr = Π = P/ω:
        #   dP/dr = d(ω·f^{-Δ})/dr · Π̃ + ω·f^{-Δ}·dΠ̃/dr 
        #           - d(ω·Δ·(f'/f))/dr · Φ - ω·Δ·(f'/f)·dΦ/dr
        #
        # For simplicity, use the direct transformation:
        #   dP/dr = ω · dΠ/dr + (dω/dr) · Π
        #         = ω · dΠ/dr + d·(f'/f)·ω · Π
        #         = ω · (dΠ/dr + d·(f'/f)·Π)
        #
        # where dΠ/dr comes from dΠ̃/dr via:
        #   dΠ/dr = f^{-Δ}·(dΠ̃/dr - Δ·(f'/f)·Π̃) - Δ·(f'/f)·dΦ/dr - Δ·(f'/f)'·Φ
        #
        # But dΦ/dr = Π (from backbone), so this simplifies.
        
        # Direct approach: transform dΠ̃/dr to dΠ/dr, then to dP/dr
        d = self.d
        
        # dΠ/dr from dΠ̃/dr:
        # Π̃ = f^Δ·(Π + Δ·(f'/f)·Φ)
        # dΠ̃/dr = Δ·(f'/f)·f^Δ·(Π + Δ·(f'/f)·Φ) + f^Δ·(dΠ/dr + Δ·(f'/f)·dΦ/dr + Δ·(f'/f)'·Φ)
        #       = Δ·(f'/f)·Π̃ + f^Δ·dΠ/dr + f^Δ·Δ·(f'/f)·Π + f^Δ·Δ·(f'/f)'·Φ
        #
        # Solving for dΠ/dr:
        # f^Δ·dΠ/dr = dΠ̃/dr - Δ·(f'/f)·Π̃ - f^Δ·Δ·(f'/f)·Π - f^Δ·Δ·(f'/f)'·Φ
        # dΠ/dr = f^{-Δ}·(dΠ̃/dr - Δ·(f'/f)·Π̃) - Δ·(f'/f)·Π - Δ·(f'/f)'·Φ
        
        # Compute (f'/f)' = d/dr[f'/f]
        # For planar AdS: f'/f = 1, so (f'/f)' = 0
        # For HSV: f'/f = -p/[(1-p)r], so (f'/f)' = p/[(1-p)r²]
        if hasattr(self.geometry, 'p') and self.geometry.p > 0:
            p = self.geometry.p
            r_safe = r.clamp(min=1e-10)
            if r_safe.ndim < phi.ndim:
                r_safe = r_safe.view(*([1] * (phi.ndim - r_safe.ndim)), *r_safe.shape).expand_as(phi)
            f_prime_over_f_deriv = p / ((1 - p) * r_safe * r_safe)
        else:
            f_prime_over_f_deriv = torch.zeros_like(phi)
        
        dPi_tilde_dr = V.pi_tilde
        f_neg_delta = f ** (-delta)
        
        dPi_dr = (f_neg_delta * (dPi_tilde_dr - delta * f_prime_over_f * pi_tilde) 
                  - delta * f_prime_over_f * pi 
                  - delta * f_prime_over_f_deriv * phi)
        
        # dP/dr = ω·dΠ/dr + (dω/dr)·Π = ω·dΠ/dr + d·(f'/f)·ω·Π
        dP_dr = omega * dPi_dr + d * f_prime_over_f * omega * pi
        
        return dP_dr

    def _omega_fn(self, r: Tensor) -> Tensor:
        """Get slice weight ω(r) = f(r)^d for symplectic integration."""
        return self.geometry.slice_weight(r)

    # =========================================================================
    # Sampling (Document Algorithm 2)
    # =========================================================================

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """
        Document Algorithm 2: Deterministic bulk→boundary generation.

        1. Sample S(r_IR) ~ μ_IR
        2. Integrate ODE dS/dr = V_KG(S,r) + R_θ(S,r) from r_IR to r_UV
        3. Return y_gen = Readout(S(r_UV))
        
        Supports both standard (RK4, etc.) and symplectic (LEAPFROG, IMPLICIT_MIDPOINT) solvers.

        Args:
            batch_size: Number of samples
            generator: Optional random generator

        Returns:
            Generated samples
        """
        state_ir = self.sample_ir_prior(batch_size, generator=generator)
        
        # Check if using symplectic solver
        is_symplectic = self.integrator_config.solver in {
            ODESolver.LEAPFROG, ODESolver.IMPLICIT_MIDPOINT
        }
        
        if is_symplectic:
            # Use symplectic integration
            # _symplectic_force calls self.velocity() which works with any residual type
            state_uv = integrate_ode_general(
                drift_fn=self.velocity,
                y0=state_ir,
                r0=self.r_ir,
                r1=self.r_uv,
                config=self.integrator_config,
                force_fn=self._symplectic_force,
                omega_fn=self._omega_fn,
                delta=self.bulk_field.deltas[0],  # Use first Delta value
                geometry=self.geometry,
            )
        else:
            # Use standard integration
            state_uv = integrate_ode(
                drift_fn=self.velocity,
                y0=state_ir,
                r0=self.r_ir,
                r1=self.r_uv,
                config=self.integrator_config,
            )

        y_gen = self.readout(state_uv)

        if self.data_mean is not None and self.data_std is not None:
            y_gen = y_gen * self.data_std + self.data_mean

        return y_gen

    # =========================================================================
    # Exact Log-Likelihood and BPD (Bits Per Dimension)
    # =========================================================================

    def compute_log_likelihood(
        self,
        data: Tensor,
        n_hutchinson_samples: int = 1,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute exact log-likelihood via the change of variables formula.

        For continuous normalizing flows:
            log p(x) = log p(z) - ∫_{r_UV}^{r_IR} div(v(S, r)) dr

        We integrate BACKWARD from UV (data) to IR (prior) while accumulating
        the divergence of the velocity field using the Hutchinson trace estimator.

        Args:
            data: Input data tensor (B, ...) at UV boundary
            n_hutchinson_samples: Number of random vectors for trace estimation
            rtol: Relative tolerance for ODE solver (unused, fixed-step)
            atol: Absolute tolerance for ODE solver (unused, fixed-step)

        Returns:
            log_likelihood: (B,) log p(x) for each sample
            log_prior: (B,) log p(z) at IR
            log_det_jacobian: (B,) accumulated divergence integral
        """
        device = data.device
        dtype = data.dtype
        
        # Normalize data if configured
        if self.data_mean is not None and self.data_std is not None:
            data = (data - self.data_mean.to(device)) / self.data_std.to(device)

        # Lift data to UV bulk state
        state_uv = self.lift_data(data)
        
        # Flatten state for integration
        phi_flat = state_uv.phi_tilde.view(data.shape[0], -1)
        pi_flat = state_uv.pi_tilde.view(data.shape[0], -1)
        state_flat = torch.cat([phi_flat, pi_flat], dim=1)  # (B, 2D)
        
        B, D = state_flat.shape
        half_D = D // 2

        # Integration parameters (backward: UV -> IR)
        n_steps = self.integrator_config.n_steps
        r_start = self.r_uv
        r_end = self.r_ir
        dr = (r_end - r_start) / n_steps  # Negative step (backward)

        # Accumulated log det Jacobian
        log_det_jacobian = torch.zeros(B, device=device, dtype=dtype)

        # Current state
        state_current = state_flat.clone()

        # Fixed-step backward integration with divergence accumulation
        for step in range(n_steps):
            r = r_start + step * dr
            r_tensor = torch.tensor(r, device=device, dtype=dtype)

            # Enable gradients for divergence computation
            state_current = state_current.detach().requires_grad_(True)

            # Reshape for velocity computation
            phi_tilde = state_current[:, :half_D].view_as(state_uv.phi_tilde)
            pi_tilde = state_current[:, half_D:].view_as(state_uv.pi_tilde)
            current_bulk_state = BulkState(phi_tilde=phi_tilde, pi_tilde=pi_tilde)

            # Compute velocity
            vel = self.velocity(current_bulk_state, r_tensor)
            vel_flat = torch.cat([
                vel.phi_tilde.view(B, -1),
                vel.pi_tilde.view(B, -1)
            ], dim=1)

            # Compute divergence using Hutchinson trace estimator
            div_estimate = torch.zeros(B, device=device, dtype=dtype)
            
            for _ in range(n_hutchinson_samples):
                # Random vector for Hutchinson estimator
                epsilon = torch.randn_like(state_current)
                
                # Compute ε^T (∂v/∂z) using vector-Jacobian product
                # vjp computes: ε^T J where J = ∂v/∂z
                vjp_result = torch.autograd.grad(
                    outputs=vel_flat,
                    inputs=state_current,
                    grad_outputs=epsilon,
                    create_graph=False,
                    retain_graph=True,
                )[0]
                
                # ε^T J ε gives trace estimate
                div_estimate = div_estimate + (epsilon * vjp_result).sum(dim=1)
            
            div_estimate = div_estimate / n_hutchinson_samples

            # Accumulate log det Jacobian
            # For CNF backward pass: we accumulate the divergence along the path
            # The divergence is negative when flow contracts (prior → data)
            log_det_jacobian = log_det_jacobian + div_estimate * abs(dr)

            # Euler step (can extend to RK4 if needed)
            with torch.no_grad():
                state_current = state_current + vel_flat * dr

        # Final state at IR
        state_ir_flat = state_current.detach()
        phi_ir = state_ir_flat[:, :half_D].view_as(state_uv.phi_tilde)
        pi_ir = state_ir_flat[:, half_D:].view_as(state_uv.pi_tilde)

        # Compute log prior probability at IR
        log_prior = self._compute_log_prior(phi_ir, pi_ir)

        # Log likelihood via change of variables:
        # log p(x) = log p(z) + log|det(dz/dx)|
        # where log|det(dz/dx)| = -∫ div(v) dr (negative of accumulated divergence)
        # Since we accumulated +div*|dr|, we need to add it (not subtract)
        # to get the correct sign for the Jacobian term
        log_likelihood = log_prior + log_det_jacobian

        return log_likelihood, log_prior, log_det_jacobian

    def _compute_log_prior(self, phi_ir: Tensor, pi_ir: Tensor) -> Tensor:
        """
        Compute log probability under the IR Gaussian prior.

        Document Section 7 (Sobolev-type Gaussian):
            p(z) = N(z | 0, Σ) where Σ = diag(σ²(λ))
            σ²(λ) = c(1+λ)^{-s}

        Args:
            phi_ir: Field at IR boundary (B, ...)
            pi_ir: Momentum at IR boundary (B, ...)

        Returns:
            log_prob: (B,) log p(z) for each sample
        """
        device = phi_ir.device
        dtype = phi_ir.dtype
        cfg = self.config
        B = phi_ir.shape[0]

        phi_flat = phi_ir.view(B, -1)
        pi_flat = pi_ir.view(B, -1)

        # Get eigenvalues for Sobolev weighting
        if cfg.prior_s_phi != 0.0 or cfg.prior_s_pi != 0.0:
            if self.slice_op is not None and hasattr(self.slice_op, "diag_eigs"):
                eigs = self.slice_op.diag_eigs(device=device, dtype=dtype)
                if eigs is not None:
                    eigs_flat = eigs.view(-1)
                    
                    # Variance per mode: σ²(λ) = c(1+λ)^{-s}
                    var_phi = cfg.prior_c_phi * (1.0 + eigs_flat) ** (-cfg.prior_s_phi)
                    var_pi = cfg.prior_c_pi * (1.0 + eigs_flat) ** (-cfg.prior_s_pi)

                    # Handle channel structure
                    n_spatial = eigs_flat.numel()
                    n_phi_total = phi_flat.shape[1]
                    n_repeats = n_phi_total // n_spatial if n_spatial > 0 else 1
                    
                    if n_repeats > 1:
                        var_phi = var_phi.repeat(n_repeats)
                        var_pi = var_pi.repeat(n_repeats)
                    
                    # Ensure shapes match
                    var_phi = var_phi[:n_phi_total]
                    var_pi = var_pi[:pi_flat.shape[1]]

                    # Log probability of Gaussian: -0.5 * (x²/σ² + log(2πσ²))
                    log_prob_phi = -0.5 * (
                        (phi_flat ** 2) / var_phi.unsqueeze(0) 
                        + torch.log(2 * math.pi * var_phi.unsqueeze(0))
                    ).sum(dim=1)
                    
                    log_prob_pi = -0.5 * (
                        (pi_flat ** 2) / var_pi.unsqueeze(0)
                        + torch.log(2 * math.pi * var_pi.unsqueeze(0))
                    ).sum(dim=1)

                    return log_prob_phi + log_prob_pi

        # Fallback: uniform variance Gaussian
        var_phi = cfg.prior_c_phi
        var_pi = cfg.prior_c_pi

        D_phi = phi_flat.shape[1]
        D_pi = pi_flat.shape[1]

        log_prob_phi = -0.5 * (
            (phi_flat ** 2).sum(dim=1) / var_phi
            + D_phi * math.log(2 * math.pi * var_phi)
        )
        log_prob_pi = -0.5 * (
            (pi_flat ** 2).sum(dim=1) / var_pi
            + D_pi * math.log(2 * math.pi * var_pi)
        )

        return log_prob_phi + log_prob_pi

    def compute_bpd(
        self,
        data: Tensor,
        n_hutchinson_samples: int = 1,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute bits per dimension (BPD) for exact likelihood evaluation.

        BPD = -log₂ p(x) / D = -log p(x) / (D × log(2))

        Lower BPD indicates higher likelihood (better model fit).

        Note: For phase-space flows, we compute BPD in the latent space.
        The dimension D is the full phase space dimension (φ + π).

        Args:
            data: Input data tensor (B, ...)
            n_hutchinson_samples: Number of random vectors for trace estimation

        Returns:
            bpd: (B,) bits per dimension for each sample
            info: Dict with 'log_likelihood', 'log_prior', 'log_det_jacobian'
        """
        log_likelihood, log_prior, log_det_jacobian = self.compute_log_likelihood(
            data, n_hutchinson_samples=n_hutchinson_samples
        )

        # Compute phase space dimension (φ + π)
        # This is the dimension over which log_prior is computed
        state_uv = self.lift_data(data[:1])  # Get shape from single sample
        D_phi = state_uv.phi_tilde[0].numel()
        D_pi = state_uv.pi_tilde[0].numel()
        D_phase_space = D_phi + D_pi
        
        # Also store data dimension for reference
        D_data = data[0].numel()

        # BPD = -log p(x) / (D × log(2))
        # Use phase space dimension for consistency with log_prior
        bpd = -log_likelihood / (D_phase_space * math.log(2))

        info = {
            "log_likelihood": log_likelihood,
            "log_prior": log_prior,
            "log_det_jacobian": log_det_jacobian,
            "data_dimension": D_data,
            "phase_space_dimension": D_phase_space,
        }

        return bpd, info

    # =========================================================================
    # Training Loss (Document Algorithm 1, Section 9, eq fm-loss)
    # =========================================================================

    def compute_losses(
        self,
        batch: Tensor,
        step: int = 0,
        epoch: int = 0,
    ) -> Dict[str, Tensor]:
        """
        Bulk flow matching loss (Document Algorithm 1, eq fm-loss).

        DOCUMENT-FAITHFUL PATH-DEPENDENT LOSS:

        HERMITE path:
            - R_θ = (0, N_θ) - residual only affects dΠ̃/dr
            - The Hermite path has Π̃_t = ∂_r Φ̃_t by construction
            - With backbone enabled (backbone_scale=1.0):
              * U_t.phi = dr × Π̃_t (from Hermite target velocity)
              * V_KG^t.phi = dr × Π̃_t (backbone's dΦ̃/dr = Π̃)
              * target_res_phi = U_t.phi - V_KG^t.phi = 0 ✓
            - Only train pi component, as phi residual is always 0

        LINEAR path:
            - R_θ = (N_θ^φ, N_θ^π) - full residual
            - Both components trained

        Document eq (fm-loss):
            L_FM(θ) = E[||R_θ^t(S_t, t) - (U_t - V_KG^t(S_t, t))||²_{g_{r(t)}}]

        Args:
            batch: Input data batch
            step: Current training step
            epoch: Current epoch

        Returns:
            Dict with 'total_loss', 'loss_dphi', 'loss_dpi', etc.
        """
        B = batch.shape[0]
        device = batch.device
        dtype = batch.dtype

        # Normalize data if configured
        if self.data_mean is not None and self.data_std is not None:
            batch = (batch - self.data_mean.to(device)) / self.data_std.to(device)

        # Sample endpoints (Document Algorithm 1)
        S0 = self.sample_ir_prior(B)
        S0 = BulkState(
            phi_tilde=S0.phi_tilde.to(device=device, dtype=dtype),
            pi_tilde=S0.pi_tilde.to(device=device, dtype=dtype),
        )

        # S_1 = Lift(y)
        if self.encoding_type in [
            FieldEncodingType.NONE,
            FieldEncodingType.HOLOGRAPHIC_GRIDFREE,
        ]:
            y = batch.view(B, -1)
        elif self.encoding_type == FieldEncodingType.SPECTRAL:
            y = batch.view(B, -1)
        else:
            y = batch
        S1 = self.lift_data(y)

        # Sample t ~ Uniform[0, 1]
        t = torch.rand(B, device=device, dtype=dtype)

        # Path interpolation and target velocity
        if self.is_hermite_path:
            S_t, U_t = self.hermite_coupling.state_and_target_velocity(S0, S1, t)
        else:
            S_t, U_t = linear_path_interpolation(S0, S1, t)

        # Physical radius at time t
        r_t = self._r_ir + t * (self._r_uv - self._r_ir)
        dr = float(self._r_uv - self._r_ir)

        # Backbone velocity - skip computation when backbone_scale=0
        if self.config.backbone_scale == 0.0:
            V_KG = BulkState(
                phi_tilde=torch.zeros_like(S_t.phi_tilde),
                pi_tilde=torch.zeros_like(S_t.pi_tilde),
            )
        else:
            V_KG = self.backbone(S_t, r_t)

        # Predicted residual depends on path type and residual type
        if self.residual_type == ResidualType.POTENTIAL:
            # Potential-based residual: force = -ω ∂U_θ/∂Φ
            omega = self.geometry.slice_weight(r_t)
            force = self.residual.compute_force(S_t.phi_tilde, t, omega)
            pred_res_phi = torch.zeros_like(S_t.phi_tilde)
            pred_res_pi = dr * force
        elif self.is_hermite_path:
            n_pi = self.residual(S_t.phi_tilde, S_t.pi_tilde, t)
            pred_res_phi = torch.zeros_like(S_t.phi_tilde)
            pred_res_pi = dr * n_pi
        else:
            n_phi, n_pi = self.residual(S_t.phi_tilde, S_t.pi_tilde, t)
            pred_res_phi = dr * n_phi
            pred_res_pi = dr * n_pi

        return compute_flow_matching_loss(
            S_t=S_t,
            U_t=U_t,
            V_KG=V_KG,
            pred_res_phi=pred_res_phi,
            pred_res_pi=pred_res_pi,
            r_t=r_t,
            dr=dr,
            backbone_scale=self.config.backbone_scale,
            geometry=self.geometry,
            is_hermite_path=self.is_hermite_path,
            use_omega_weighting=self.config.use_omega_weighting,
        )


# =============================================================================
# Backwards-compatible aliases
# =============================================================================

HolographicGenerator = UVStabilizedFlowMatchingModel