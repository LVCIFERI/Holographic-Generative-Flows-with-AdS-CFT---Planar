"""
config.py

Configuration Classes for AdS/CFT Flow Matching Framework
=========================================================

This module consolidates ALL configuration dataclasses and enumerations
for the framework into a single, well-organized location.

Organization
------------
1. Enumerations (immutable choices)
   - SliceGeometry: AdS foliation types
   - PathType: Flow matching path types
   - FieldEncodingType: Point-to-field encoding methods
   - DatasetType: Available dataset types
   - ODESolver: ODE integration methods

2. Core Configuration Dataclasses
   - GeometryConfig: AdS geometry parameters
   - DiscretizationConfig: Grid/spectral discretization
   - PathConfig: Flow matching path configuration
   - IntegratorConfig: ODE integration settings

3. Model Configuration
   - FlowModelConfig: Main model hyperparameters

4. Training Configuration
   - TrainerConfig: Training loop settings
   - ExperimentConfig: Complete experiment specification
   - BaselineConfig: Baseline model settings

Document References
-------------------
- Section 2: Geometry (SliceGeometry, GeometryConfig)
- Section 3: Discretization (DiscretizationConfig)
- Section 4: Path types and residual (PathConfig)
- Section 5: Field encoding (FieldEncodingType)
- Section 7: IR prior (prior_* parameters in configs)
- Section 9: Path schedules (PathType)
- Algorithm 1: Training (TrainerConfig, ExperimentConfig)
- Algorithm 2: Sampling/ODE integration (IntegratorConfig, ODESolver)

Design Principles
-----------------
1. Immutability: Use frozen=True for configs that shouldn't change after creation
2. Defaults: Provide sensible defaults that match document recommendations
3. Validation: Use __post_init__ for validation where needed
4. Documentation: Every field has a docstring explaining its purpose
5. Type Safety: Strong typing with Optional for nullable fields
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Optional, Tuple, Union

import torch


# =============================================================================
# Enumerations
# =============================================================================


class SliceGeometry(str, Enum):
    """
    Supported AdS foliations / slice geometries.

    Document Definition 2.1 (Common warped-product slicings):

    Single-warp foliations ds² = dr² + f(r)² ĝ_ab dx^a dx^b:
        - PLANAR: Σ ≅ ℝ^d with flat ĝ, f(r) = e^r

    Ablation:
        - FLAT: No AdS curvature, f(r) = 1 (for testing)

    Hyperscaling-violating:
        - HYPERSCALING_VIOLATING: f(r) = [(1-p)r]^(-p/(1-p)), p ∈ [0, 1)
    """

    PLANAR = "planar"
    FLAT = "flat"  # Ablation: flat Euclidean space, no AdS
    HYPERSCALING_VIOLATING = "planar_hsv"  # Hyperscaling-violating: f(r) = [(1-p)r]^(-p/(1-p))


class PathType(str, Enum):
    """
    Path type for flow matching interpolation.

    Document Section 9:
        - LINEAR: S_t = (1-t)S_0 + tS_1, U_t = S_1 - S_0 (robust practical default)
        - HERMITE: AdS-affine cubic Hermite interpolation (eq rectified-path)
                   Π̃_t = ∂_r Φ̃_t by construction (document-faithful)
        - QUADRATIC: Quadratic Bezier path with midpoint interpolation
    """

    LINEAR = "linear"
    ADS_AFFINE_HERMITE = "hermite"
    QUADRATIC = "quadratic"


class FieldEncodingType(str, Enum):
    """
    Type of point-to-field encoding.

    Options:
        - NONE: No encoding (point data or native fields)
        - GAUSSIAN: Legacy Gaussian blob encoding (ad-hoc)
        - HOLOGRAPHIC: AdS/CFT bulk-boundary propagator on grid (memory heavy)
        - HOLOGRAPHIC_GRIDFREE: Grid-free holographic (limited physics)
        - SPECTRAL: Spectral/Fourier holographic (RECOMMENDED - fast + faithful)
        - IMAGE: Direct FFT for image data (images ARE boundary fields)
        - VANILLA_CNN: Vanilla CNN flow matching (no physics, raw images)

    Document Section 5:
        The bulk-boundary propagator K_Δ(z, x; x') encodes boundary sources
        into genuine bulk field configurations.
    """

    NONE = "none"
    GAUSSIAN = "gaussian"
    HOLOGRAPHIC = "holographic"
    HOLOGRAPHIC_GRIDFREE = "holographic_gridfree"
    SPECTRAL = "spectral"
    IMAGE = "image"
    VANILLA_CNN = "vanilla_cnn"


class DatasetType(str, Enum):
    """
    Available toy dataset types.

    Organized by geometry:

    Planar (ℝ^d):
        - CHECKERBOARD: 2D checkerboard pattern
        - GAUSSIAN_MIXTURE: Mixture of Gaussians
        - SWISS_ROLL: 3D Swiss roll manifold
        - TWO_MOONS: Sklearn two moons
        - CONCENTRIC_CIRCLES: Nested circles
        - PINWHEEL: Spiral clusters
    """

    # Planar (ℝ^d)
    CHECKERBOARD = "checkerboard"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    SWISS_ROLL = "swiss_roll"
    TWO_MOONS = "two_moons"
    CONCENTRIC_CIRCLES = "concentric_circles"
    PINWHEEL = "pinwheel"


class ODESolver(str, Enum):
    """
    Available ODE solvers for integration.

    Document Algorithm 2 specifies integration from r_IR to r_UV.

    Options:
        - EULER: First-order explicit Euler (fast, low accuracy)
        - HEUN: Second-order Heun's method (explicit trapezoidal)
        - MIDPOINT: Second-order midpoint method
        - RK4: Fourth-order Runge-Kutta (default, recommended)
        - LEAPFROG: Symplectic Störmer-Verlet (energy-conserving, separable H)
        - IMPLICIT_MIDPOINT: Symplectic implicit midpoint (general Hamiltonian)
    """

    EULER = "euler"
    HEUN = "heun"
    MIDPOINT = "midpoint"
    RK4 = "rk4"
    # Symplectic integrators (for Hamiltonian systems)
    LEAPFROG = "leapfrog"
    IMPLICIT_MIDPOINT = "implicit_midpoint"


class ResidualType(str, Enum):
    """
    Type of residual network parameterization.
    
    DIRECT: The residual network outputs a direct correction to momentum.
        R_θ(S, r) = (0, N_θ(S, r))
        Simple and flexible but not symplectic.
    
    POTENTIAL: The residual is derived from a learned potential U_θ(Φ, r).
        R_θ(S, r) = (0, -ω(r) ∂U_θ/∂Φ)
        Keeps full dynamics Hamiltonian, enabling symplectic integration.
    
    For symplectic integrators (LEAPFROG, IMPLICIT_MIDPOINT), use POTENTIAL.
    For standard integrators (RK4, etc.), DIRECT is simpler.
    """
    
    DIRECT = "direct"
    POTENTIAL = "potential"


# =============================================================================
# Core Configuration Dataclasses
# =============================================================================


@dataclass(frozen=True)
class GeometryConfig:
    """
    Configuration for an AdS foliation.

    Document Section 2: AdS foliations and warp factors.

    Parameters
    ----------
    d : int
        Slice/boundary dimension dim(Σ) = d.
    slice_geometry : SliceGeometry
        The slicing type (planar, flat, or planar_hsv).
    dtype : torch.dtype
        Data type for computations.
    device : str or torch.device
        Device for computations.
    affine_cache_size : int
        Number of points for trapezoidal cache (affine parameter computation).
    affine_cache_margin : float
        Small margin for cache boundaries.
    eps : float
        Small epsilon for numerical stability.
    r_min : float
        Default IR cutoff radius.
    r_max : float
        Default UV cutoff radius.
    """

    d: int
    slice_geometry: SliceGeometry = SliceGeometry.PLANAR
    dtype: torch.dtype = torch.float32
    device: Union[str, torch.device] = "cpu"
    affine_cache_size: int = 4096
    affine_cache_margin: float = 1e-6
    eps: float = 1e-12
    r_min: float = 0.0
    r_max: float = 8.0
    # Hyperscaling-violating parameter p ∈ [0, 1)
    # p = 1: flat space (f = 1)
    # p → 0: AdS limit (f = e^r)
    # Intermediate: f(r) = [(1-p)r]^(-p/(1-p))
    hsv_p: float = 0.5
    # Threshold for switching to exact AdS formulas near p = 0
    hsv_ads_threshold: float = 0.05
    # Propagator mode: "bessel" for closed-form conformal product, "emd" for full ODE
    hsv_propagator_mode: str = "bessel"


@dataclass(frozen=True)
class DiscretizationConfig:
    """
    Discretization choice for the slice operator.

    Document Section 3: Spatial discretization of fields on slices.

    Parameters
    ----------
    representation : str
        "spectral" (eigenbasis of -Δ_ĝ) or "physical" (coordinate grid)

    Planar / Generic FFT Grid:
        grid_shape : tuple of int
            Spatial grid dimensions
        box_lengths : tuple of float
            Physical domain size in each dimension
    """

    representation: Literal["spectral", "physical"] = "spectral"

    # Planar / generic FFT grid
    grid_shape: Optional[Tuple[int, ...]] = None
    box_lengths: Optional[Tuple[float, ...]] = None


@dataclass(frozen=True)
class PathConfig:
    """
    Configuration for flow matching path.

    Document Section 9: Bulk flow matching paths.

    Parameters
    ----------
    path_type : PathType
        LINEAR: S_t = (1-t)S_0 + tS_1, U_t = S_1 - S_0 (robust default)
        HERMITE: AdS-affine cubic Hermite (eq rectified-path, target-velocity)
    """

    path_type: PathType = PathType.LINEAR


@dataclass(frozen=True)
class IntegratorConfig:
    """
    Configuration for ODE integration.

    Document Algorithm 2: Integrate dS/dr = V_θ(S, r) from r_IR to r_UV.

    Parameters
    ----------
    solver : ODESolver
        Integration method (euler, heun, midpoint, rk4, leapfrog, implicit_midpoint)
    n_steps : int
        Number of fixed integration steps
    use_affine_stepping : bool
        If True, step uniformly in affine parameter u(r) instead of r.
        Recommended for HSV geometries where ω(r) varies wildly.
    implicit_tol : float
        Tolerance for implicit midpoint fixed-point iteration
    implicit_max_iter : int
        Maximum iterations for implicit midpoint
    """

    solver: ODESolver = ODESolver.RK4
    n_steps: int = 100
    use_affine_stepping: bool = False  # Step in u(r) instead of r
    implicit_tol: float = 1e-6  # For implicit midpoint
    implicit_max_iter: int = 10  # For implicit midpoint


# =============================================================================
# Model Configuration
# =============================================================================


@dataclass
class FlowModelConfig:
    """
    Configuration for the UV-stabilized flow matching model.

    Document references:
        - d: Boundary dimension (Section 2)
        - deltas: Conformal dimensions Δ_i > d/2 (eq mass-dimension)
        - r_ir, r_uv: Radial domain (Section 7, Section 5)
        - lift_noise_sigma: σ_L for UV lift noise (Section 5.1)
        - prior_*: IR prior parameters (Section 7)

    Parameters
    ----------
    d : int
        Boundary/slice dimension.
    deltas : tuple of float
        Conformal dimensions for each channel. Must satisfy Δ_i > d/2.
    discretization : DiscretizationConfig
        Spatial discretization settings.
    geometry : GeometryConfig, optional
        AdS geometry configuration. Created automatically if None.
    r_ir : float
        IR cutoff radius (interior boundary).
    r_uv : float
        UV cutoff radius (boundary).
    lift_noise_sigma : float
        Standard deviation for UV lift noise ξ (Section 5.1).

    IR Prior (Section 7):
        prior_c_phi : float
            Scale for Φ̃ prior covariance.
        prior_s_phi : float
            Sobolev smoothing exponent for Φ̃.
        prior_c_pi : float
            Scale for Π̃ prior covariance.
        prior_s_pi : float
            Sobolev smoothing exponent for Π̃.

    Document-Faithful Defaults:
        hermite_pi_scale : float
            Scale for Π̃ values in Hermite construction.
        backbone_scale : float
            Backbone contribution (1.0 = full backbone, required for Hermite).

    Residual Network (Section 4.2):
        residual_hidden : int
            Hidden dimension for residual network.
        residual_depth : int
            Number of residual blocks.
        residual_time_emb_dim : int
            Time embedding dimension.

    Path Configuration (Section 9):
        path_config : PathConfig
            Flow matching path settings.
        integrator : IntegratorConfig
            ODE integration settings.

    Point-to-Field Encoding:
        use_field_encoding : bool
            Enable legacy Gaussian blob encoding.
        use_spectral_encoding : bool
            Enable spectral holographic encoding (RECOMMENDED).
        use_holographic_encoding : bool
            Enable grid-free holographic encoding.
        use_holographic_grid : bool
            Enable grid-based holographic encoding.
        use_image_encoding : bool
            Enable image encoding for MNIST/CIFAR.
        image_shape : tuple, optional
            (C, H, W) for image data.

    Spectral Encoding Parameters:
        spectral_n_modes : int
            Number of Fourier modes per dimension.
        spectral_decode_resolution : int
            Output resolution for position decoding.
        spectral_decode_method : str
            Decoding method ('softargmax', 'phase', 'density').
        spectral_random_feature_scale : float
            Scale for random k-vectors (d > 3).

    Grid-Based Encoding Parameters:
        field_grid_size : (int, int)
            Spatial resolution H×W.
        field_domain : tuple of float
            Domain bounds (x_min, x_max, y_min, y_max).
        field_sigma : float
            Gaussian blob width.
        field_temperature : float
            Softmax temperature for decoding.
        holographic_regularization : float
            Stability parameter for propagator.
        holographic_normalize : bool
            Normalize propagator integral.

    Laplacian Type:
        laplacian_type : str
            'diagonal' - FFT-based spectral Laplacian (exact for planar).
    """

    # Core parameters
    d: int
    deltas: Tuple[float, ...]

    # Discretization
    discretization: DiscretizationConfig

    # Geometry
    geometry: Optional[GeometryConfig] = None

    # Radial domain
    r_ir: float = 0.0
    r_uv: float = 1.0

    # UV lift noise (Section 5.1)
    lift_noise_sigma: float = 0.1

    # IR prior (Section 7)
    # Sobolev-type: σ²(λ) = c(1+λ)^{-s}
    prior_c_phi: float = 1.0
    prior_s_phi: float = 1.0
    prior_c_pi: float = 0.55  # Balanced for Hermite
    prior_s_pi: float = 1.0

    # Document-faithful defaults
    hermite_pi_scale: float = 1.0
    backbone_scale: float = 1.0  # Required for Hermite path

    # Residual network (Section 4.2)
    # These are optional overrides. If None, uses cnn_*/mlp_* defaults based on network type.
    residual_hidden: Optional[int] = None
    residual_depth: Optional[int] = None
    residual_time_emb_dim: int = 128
    residual_type: ResidualType = ResidualType.DIRECT  # DIRECT or POTENTIAL (symplectic)
    
    # CNN defaults (for spectral/image/field encoding)
    cnn_hidden: int = 64
    cnn_depth: int = 3
    
    # MLP defaults (for point data without grid structure)
    mlp_hidden: int = 512
    mlp_depth: int = 6

    # Path type (Section 9)
    path_config: PathConfig = field(default_factory=PathConfig)
    
    # Radial Hermite: use t-parameterised Hermite (no ω factors in tangents)
    # Stable for any geometry, auto-enabled for HSV
    use_radial_hermite: bool = False
    
    # Loss weighting: whether to multiply MSE by slice volume ω = f(r)^d
    # True: geometrically correct L² norm on slice (default)
    # False: unweighted MSE (numerical proxy, treats all radii equally)
    use_omega_weighting: bool = True

    # ODE integration (Algorithm 2)
    integrator: IntegratorConfig = field(default_factory=IntegratorConfig)

    # Point-to-field encoding options
    use_field_encoding: bool = False
    use_spectral_encoding: bool = False
    use_holographic_encoding: bool = False
    use_holographic_grid: bool = False
    use_image_encoding: bool = False
    image_shape: Optional[Tuple[int, int, int]] = None

    # Spectral encoding parameters
    spectral_n_modes: int = 16
    spectral_decode_resolution: int = 64
    spectral_decode_method: str = "softargmax"
    spectral_random_feature_scale: float = 1.0
    
    # HSV propagator mode for spectral encoding
    # "bessel": Closed-form Bessel solution (conformal product, fast)
    # "emd": Full EMD ODE integration (exact, slower)
    hsv_propagator_mode: str = "bessel"

    # Shared field encoding parameters
    field_grid_size: Tuple[int, int] = (32, 32)
    field_domain: Tuple[float, ...] = (-4.0, 4.0, -4.0, 4.0)
    field_sigma: float = 0.3
    field_temperature: float = 0.1

    # Holographic encoding specific
    holographic_regularization: float = 1e-2
    holographic_normalize: bool = True

    # Laplacian type selection
    laplacian_type: str = "diagonal"


# =============================================================================
# Training Configuration
# =============================================================================


@dataclass
class TrainerConfig:
    """
    Configuration for the training process.

    Document Algorithm 1 specifies:
        - Sampling from data distribution p_data
        - Sampling from IR prior μ_IR
        - Sampling time t ~ Uniform[0,1]
        - Gradient updates on backbone-subtracted loss

    Parameters
    ----------
    Training Duration:
        max_epochs : int
            Maximum training epochs.
        max_steps : int, optional
            If set, overrides max_epochs.

    Batch Size:
        batch_size : int
            Training batch size.

    Optimizer:
        learning_rate : float
            Initial learning rate.
        weight_decay : float
            L2 regularization strength.
        betas : (float, float)
            Adam beta parameters.
        eps : float
            Adam epsilon for numerical stability.

    Learning Rate Schedule:
        lr_scheduler : str
            Schedule type ("cosine", "constant", "warmup_cosine").
        warmup_steps : int
            Number of warmup steps.
        min_lr_ratio : float
            Minimum LR as ratio of initial LR.

    EMA (Exponential Moving Average):
        use_ema : bool
            Enable EMA for stable sampling.
        ema_decay : float
            EMA decay rate.
        ema_start_step : int
            Step to start EMA updates.
        ema_update_every : int
            Update EMA every N steps.

    Mixed Precision:
        use_amp : bool
            Enable automatic mixed precision.
        amp_dtype : str
            AMP dtype ("float16" or "bfloat16").
        grad_scaler_init_scale : float
            Initial gradient scaler scale.

    Gradient Clipping:
        max_grad_norm : float, optional
            Maximum gradient norm. None disables clipping.

    Logging and Checkpointing:
        log_every : int
            Log metrics every N steps.
        eval_every : int
            Run evaluation every N steps.
        checkpoint_every : int
            Save checkpoint every N steps.
        checkpoint_dir : str
            Directory for checkpoints.
        keep_last_n_checkpoints : int
            Number of recent checkpoints to keep.

    Data Loading:
        num_workers : int
            DataLoader worker processes.
        pin_memory : bool
            Pin memory for faster GPU transfer.
        prefetch_factor : int
            Batches to prefetch per worker.

    Device:
        device : str
            Computation device ("cuda" or "cpu").
        seed : int, optional
            Random seed for reproducibility.
    """

    # Training duration
    max_epochs: int = 1000
    max_steps: Optional[int] = None

    # Batch size
    batch_size: int = 64

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    # Learning rate schedule
    lr_scheduler: str = "cosine"
    warmup_steps: int = 1000
    min_lr_ratio: float = 0.1

    # EMA for stable sampling
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_start_step: int = 1000
    ema_update_every: int = 1

    # Mixed precision
    use_amp: bool = False
    amp_dtype: str = "float16"
    grad_scaler_init_scale: float = 65536.0

    # Gradient clipping
    max_grad_norm: Optional[float] = 1.0

    # Logging and checkpointing
    log_every: int = 100
    eval_every: int = 1000
    checkpoint_every: int = 5000
    checkpoint_dir: str = "./checkpoints"
    keep_last_n_checkpoints: int = 5

    # Reproducibility
    seed: Optional[int] = None

    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.

    This is the top-level configuration that combines all settings for
    a complete training run.

    Document references:
        - d: Boundary dimension (Section 2)
        - deltas: Conformal dimensions (eq mass-dimension)
        - r_ir, r_uv: Radial domain
        - path_type: Section 9 (linear vs hermite)

    Encoding Options:
        1. NO ENCODING (default): Points as raw data
        2. GAUSSIAN ENCODING (--use_field_encoding): Legacy Gaussian blobs
        3. HOLOGRAPHIC ENCODING (--use_holographic_encoding): Bulk-boundary propagator

    Parameters
    ----------
    Experiment Metadata:
        name : str
            Experiment name for logging.
        seed : int
            Random seed.
        output_dir : str
            Output directory for results.

    Data:
        dataset : str
            Dataset name (e.g., "checkerboard", "mnist").
        n_train_samples : int
            Training set size.
        n_val_samples : int
            Validation set size.
        tile_n : int
            Tiles per dimension for checkerboard patterns.
        checker_size : float
            Size parameter for checkerboard.

    Geometry (Section 2):
        d : int, optional
            Boundary dimension. Auto-detected from data if None.
        slice_geometry : str
            Geometry type ("planar", "flat", or "planar_hsv").

    Bulk Field (Section 5):
        deltas : tuple of float
            Conformal dimensions.

    Radial Domain:
        r_ir : float
            IR cutoff radius.
        r_uv : float
            UV cutoff radius.

    UV Lift (Section 5.1):
        lift_noise_sigma : float
            Noise scale for UV lift.

    IR Prior (Section 7):
        prior_c_phi, prior_s_phi : float
            Φ̃ prior parameters.
        prior_c_pi, prior_s_pi : float
            Π̃ prior parameters.

    Path (Section 9):
        path_type : str
            "linear" or "hermite".
        hermite_pi_scale : float
            Scale for Π̃ in Hermite.
        backbone_scale : float
            Backbone contribution (1.0 for Hermite).

    Field Encoding:
        use_field_encoding : bool
            Legacy Gaussian encoding.
        use_spectral_encoding : bool
            Spectral holographic (RECOMMENDED).
        use_holographic_encoding : bool
            Grid-free holographic.
        use_holographic_grid : bool
            Grid-based holographic.
        spectral_* : various
            Spectral encoding parameters.
        field_* : various
            Grid-based encoding parameters.
        holographic_* : various
            Holographic encoding parameters.

    Laplacian:
        laplacian_type : str
            "diagonal" - FFT-based spectral Laplacian.

    Image Encoding:
        use_image_encoding : bool
            Enable for image datasets.

    Residual Network (Section 4.2):
        residual_hidden : int
            Hidden dimension.
        residual_depth : int
            Network depth.
        residual_time_emb_dim : int
            Time embedding dimension.

    ODE Integration (Algorithm 2):
        ode_solver : str
            Solver type.
        ode_n_steps : int
            Integration steps.

    Training:
        epochs : int
            Training epochs.
        batch_size : int
            Batch size.
        lr : float
            Learning rate.
        weight_decay : float
            L2 regularization.
        warmup_steps : int
            LR warmup steps.
        max_grad_norm : float
            Gradient clipping.

    EMA:
        use_ema : bool
            Enable EMA.
        ema_decay : float
            EMA decay rate.

    Logging:
        log_every : int
            Log frequency.
        eval_every : int
            Evaluation frequency.
        checkpoint_every : int
            Checkpoint frequency.

    Visualization:
        n_viz_real, n_viz_gen : int
            Samples for visualization.
        viz_every_epochs : int
            Visualization frequency.

    Hardware:
        device : str
            Computation device.
        use_amp : bool
            Mixed precision.
        num_workers : int
            Data loading workers.
    """

    # Experiment metadata
    name: str = "ads_flow_matching"
    seed: int = 42
    output_dir: str = "./outputs"

    # Data
    dataset: str = "checkerboard"
    n_train_samples: int = 50000
    n_val_samples: int = 5000

    # Dataset-specific parameters for checkerboard
    tile_n: int = 4
    checker_size: float = 8.0

    # Geometry (Document Section 2)
    d: Optional[int] = None
    slice_geometry: str = "planar"
    # Hyperscaling-violating parameter p ∈ (0, 1]
    # p = 1: flat, p → 0: AdS limit
    hsv_p: float = 0.5
    hsv_ads_threshold: float = 0.05
    # Propagator mode: "bessel" for closed-form conformal product, "emd" for full ODE
    hsv_propagator_mode: str = "bessel"
    
    # HSV u-bounds: stable training for ALL p ∈ (0, 1]
    # The HSV coordinate u = (p*r)^{1/p} is the natural variable:
    #   - Appears directly in propagator: K̂(u,k) = (|k|u)^β K_β(|k|u)
    #   - p = 1: u = r (flat), p → 0: u = e^{-r} = z (AdS)
    # Fixing u-bounds ensures r_ir stays away from r=0 singularity for ALL p.
    # r = u^{p} / p, so r_ir > 0 always when u_uv > 0.
    hsv_use_u_bounds: bool = True
    hsv_u_uv: float = 0.1   # UV boundary (small u, near boundary)
    hsv_u_ir: float = 1.0   # IR boundary (large u, deep bulk)

    # Bulk field (Document Section 5)
    deltas: Tuple[float, ...] = (1.5, 1.5)

    # Radial domain
    r_ir: float = 0.0
    r_uv: float = 1.0

    # UV lift (Document Section 5.1)
    lift_noise_sigma: float = 0.1

    # IR prior (Document Section 7)
    prior_c_phi: float = 1.0
    prior_s_phi: float = 1.0
    prior_c_pi: float = 0.55
    prior_s_pi: float = 1.0

    # Path (Document Section 9)
    path_type: str = "hermite"
    
    # Radial Hermite: use t-parameterised Hermite instead of affine u(r)-parameterised
    # Radial Hermite has tangents m = r_span * Π̃ (no ω factors), stable for any geometry
    # Affine Hermite has tangents m = Z * ω * Π̃ (geometry-weighted)
    # Default: False for non-HSV (use affine), auto-enabled for HSV in code
    use_radial_hermite: bool = False
    
    # Loss weighting: whether to multiply MSE by slice volume ω = f(r)^d
    # True: geometrically correct L² norm on slice (default)
    # False: unweighted MSE (numerical proxy, treats all radii equally)
    use_omega_weighting: bool = True

    # Document-faithful defaults
    hermite_pi_scale: float = 1.0
    backbone_scale: float = 1.0

    # Field encoding options
    use_field_encoding: bool = False
    use_spectral_encoding: bool = False
    use_holographic_encoding: bool = False
    use_holographic_grid: bool = False

    # Spectral encoding parameters
    spectral_n_modes: int = 16
    spectral_decode_resolution: int = 64
    spectral_decode_method: str = "softargmax"
    spectral_random_feature_scale: float = 1.0

    # Shared field encoding parameters
    field_grid_size: Tuple[int, int] = (32, 32)
    field_domain: Tuple[float, ...] = (-4.0, 4.0, -4.0, 4.0)
    field_sigma: float = 0.3
    field_temperature: float = 0.1

    # Holographic encoding specific
    holographic_regularization: float = 1e-2
    holographic_normalize: bool = True

    # Laplacian type selection
    laplacian_type: str = "diagonal"

    # Image encoding
    use_image_encoding: bool = False
    
    # Vanilla CNN (no physics, raw images with CNN architecture)
    use_vanilla_cnn: bool = False

    # Residual network (Document Section 4.2)
    # These are optional overrides. If None, uses cnn_*/mlp_* defaults based on network type.
    residual_hidden: Optional[int] = None
    residual_depth: Optional[int] = None
    residual_time_emb_dim: int = 128
    residual_type: str = "direct"  # "direct" or "potential" (symplectic)
    
    # CNN defaults (for spectral/image/field encoding)
    cnn_hidden: int = 64
    cnn_depth: int = 3
    
    # MLP defaults (for point data without grid structure)
    mlp_hidden: int = 512
    mlp_depth: int = 6

    # ODE integration (Document Algorithm 2)
    ode_solver: str = "rk4"
    ode_n_steps: int = 100
    use_affine_stepping: bool = False  # Step in affine u(r) for HSV
    implicit_tol: float = 1e-6  # For implicit midpoint
    implicit_max_iter: int = 10  # For implicit midpoint

    # Training
    epochs: int = 5000
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9999

    # Logging
    log_every: int = 100
    eval_every: int = 1000
    checkpoint_every: int = 5000

    # Visualization
    n_viz_real: int = 10000
    n_viz_gen: int = 10000
    viz_every_epochs: int = 10

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = False
    num_workers: int = 4


@dataclass
class BaselineConfig:
    """
    Configuration for baseline flow matching models.

    Used for fair comparison experiments.

    Parameters
    ----------
    Experiment:
        name : str
            Experiment name.
        seed : int
            Random seed.
        output_dir : str
            Output directory.

    Baseline Type:
        baseline_type : str
            "mlp" (simple) or "spectral" (fair comparison).

    Data:
        dataset : str
            Dataset name.
        n_train_samples : int
            Training samples.
        n_val_samples : int
            Validation samples.

    Model (MLP baseline):
        hidden : int
            Hidden dimension.
        depth : int
            Network depth.
        time_emb_dim : int
            Time embedding dimension.

    Spectral Baseline:
        spectral_n_modes : int
            Fourier modes.
        spectral_deltas : tuple of float
            Conformal dimensions.
        spectral_geometry : str
            Geometry type.

    Training:
        epochs : int
            Training epochs.
        batch_size : int
            Batch size.
        lr : float
            Learning rate.
        weight_decay : float
            L2 regularization.
        max_grad_norm : float
            Gradient clipping.

    EMA:
        use_ema : bool
            Enable EMA.
        ema_decay : float
            Decay rate.
        ema_start_step : int
            Start step.

    ODE:
        n_ode_steps : int
            Integration steps.

    Logging:
        log_every : int
            Log frequency.
        eval_every : int
            Evaluation frequency.
        viz_every_epochs : int
            Visualization frequency.
        n_viz_samples : int
            Samples for visualization.

    Device:
        device : str
            Computation device.
    """

    # Experiment
    name: str = "baseline_flow_matching"
    seed: int = 42
    output_dir: str = "outputs"

    # Baseline type
    baseline_type: str = "spectral"

    # Data
    dataset: str = "checkerboard"
    n_train_samples: int = 50000
    n_val_samples: int = 10000

    # Model (MLP baseline)
    hidden: int = 256
    depth: int = 4
    time_emb_dim: int = 128

    # Spectral baseline options
    spectral_n_modes: int = 16
    spectral_deltas: Tuple[float, ...] = (1.5, 2.5)
    spectral_geometry: str = "planar"

    # Training
    epochs: int = 100
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-5

    # Gradient clipping
    max_grad_norm: float = 1.0

    # EMA
    use_ema: bool = False
    ema_decay: float = 0.9999
    ema_start_step: int = 100

    # ODE
    n_ode_steps: int = 100

    # Logging
    log_every: int = 100
    eval_every: int = 1000
    viz_every_epochs: int = 10
    n_viz_samples: int = 10000

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Note: BulkState is defined in networks.py to keep data structures with their
# operations. Import it from there: from ads_cft.networks import BulkState
# =============================================================================
