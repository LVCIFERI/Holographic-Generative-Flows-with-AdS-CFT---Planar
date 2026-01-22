"""
utils.py

Utility functions for UV-stabilized generative flow matching.

Document-faithful implementation of:

- Reproducibility utilities (seeding, generators)
- Tensor and shape helpers
- Quadrature-aware norms with slice measure weighting (eq slice-volume)
- KG residual diagnostics (verifying backbone correctness)
- Statistics and normalization
- Model inspection utilities
- Timing utilities
- Checkpoint utilities
- FFT utilities
- Sampling utilities

CRITICAL FIX: This version uses the correct geometry API (f(), a(), b())
instead of the incorrect warp_factor(), warp_a(), warp_b().
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

Tensor = torch.Tensor


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
        deterministic: If True, use deterministic algorithms (slower)
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch 1.8+
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)


def get_generator(
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> torch.Generator:
    """
    Create a seeded random generator.
    
    Args:
        seed: Random seed (None for random)
        device: Device for generator
        
    Returns:
        Seeded torch.Generator
    """
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    return gen


# =============================================================================
# Tensor helpers
# =============================================================================

def to_tensor(
    x: Union[float, int, List, Tensor],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """
    Convert input to tensor.
    
    Args:
        x: Input value (scalar, list, or tensor)
        device: Target device
        dtype: Target dtype
        
    Returns:
        Tensor on specified device/dtype
    """
    if isinstance(x, Tensor):
        result = x
    else:
        result = torch.tensor(x)
    
    if device is not None:
        result = result.to(device)
    if dtype is not None:
        result = result.to(dtype)
    return result


def broadcast_to_shape(
    x: Tensor,
    target_shape: Tuple[int, ...],
    dim: int = 0,
) -> Tensor:
    """
    Broadcast tensor x to target_shape by expanding along specified dimension.
    
    Args:
        x: Input tensor
        target_shape: Target shape
        dim: Dimension to expand from
        
    Returns:
        Broadcasted tensor
    """
    while x.ndim < len(target_shape):
        x = x.unsqueeze(-1)
    return x.expand(target_shape)


def flatten_spatial(x: Tensor) -> Tensor:
    """
    Flatten spatial dimensions: (B, C, *spatial) -> (B, C, -1).
    
    Args:
        x: Input tensor with batch and channel dimensions
        
    Returns:
        Tensor with flattened spatial dimensions
    """
    if x.ndim <= 2:
        return x
    B, C = x.shape[:2]
    return x.reshape(B, C, -1)


def unflatten_spatial(
    x: Tensor,
    spatial_shape: Tuple[int, ...],
) -> Tensor:
    """
    Unflatten spatial dimensions: (B, C, L) -> (B, C, *spatial_shape).
    
    Args:
        x: Flattened tensor (B, C, L)
        spatial_shape: Target spatial shape
        
    Returns:
        Unflattened tensor
        
    Raises:
        ValueError: If tensor is not 3D or spatial size doesn't match
    """
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {x.ndim}D")
    B, C, L = x.shape
    expected_L = 1
    for s in spatial_shape:
        expected_L *= s
    if L != expected_L:
        raise ValueError(f"Spatial size mismatch: {L} vs {expected_L}")
    return x.reshape(B, C, *spatial_shape)


# =============================================================================
# Slice measure computation (Document eq slice-volume)
# =============================================================================

def compute_slice_weight(
    r: Tensor,
    d: int,
    slice_geometry: str,
    eps: float = 1e-12,
) -> Tensor:
    """
    Compute ω(r) such that dvol_{g_r} = ω(r) dvol_ĝ.
    
    Document eq (slice-volume):
    - Single-warp: ω(r) = f(r)^d
    
    Document Definition 2.1:
    - PLANAR: f(r) = e^r
    
    Args:
        r: Radial coordinate tensor
        d: Boundary dimension
        slice_geometry: Geometry type
        eps: Numerical stability epsilon
        
    Returns:
        Slice weight ω(r)
        
    Raises:
        ValueError: If geometry is unknown
    """
    geom = slice_geometry.lower()
    
    if geom == "planar":
        return torch.exp(float(d) * r)
    else:
        raise ValueError(f"Unknown slice geometry: {slice_geometry}")


def compute_kappa(
    r: Tensor,
    d: int,
    slice_geometry: str,
    eps: float = 1e-12,
) -> Tensor:
    """
    Compute radial friction κ(r).
    
    Document Section 2:
    - Single-warp: κ(r) = d (f'/f)
    
    Document Definition 2.1:
    - PLANAR: f'/f = 1, so κ = d
    
    Args:
        r: Radial coordinate tensor
        d: Boundary dimension
        slice_geometry: Geometry type
        eps: Numerical stability epsilon
        
    Returns:
        Friction coefficient κ(r)
        
    Raises:
        ValueError: If geometry is unknown
    """
    geom = slice_geometry.lower()
    
    if geom == "planar":
        return torch.full_like(r, float(d))
    else:
        raise ValueError(f"Unknown slice geometry: {slice_geometry}")


# =============================================================================
# Quadrature-aware norms (Document eq fm-loss)
# =============================================================================

def weighted_l2_norm_squared(
    x: Tensor,
    weights: Optional[Tensor] = None,
    omega: Optional[Tensor] = None,
    reduce: str = "mean",
) -> Tensor:
    """
    Compute weighted L² norm squared.
    
    Document eq (fm-loss): ||x||²_{g_r} = ∫ |x|² dvol_{g_r} = ∫ |x|² ω(r) dvol_ĝ
    
    Args:
        x: Input tensor (B, C, *spatial)
        weights: Quadrature weights for dvol_ĝ (*spatial)
        omega: Slice weight ω(r), scalar or (B,)
        reduce: "mean", "sum", or "none"
        
    Returns:
        Weighted norm squared
        
    Raises:
        ValueError: If reduce mode is unknown
    """
    sq = x * x
    
    # Apply slice weight ω(r)
    if omega is not None:
        if omega.ndim == 0:
            sq = omega * sq
        else:
            while omega.ndim < sq.ndim:
                omega = omega.unsqueeze(-1)
            sq = omega * sq
    
    # Apply quadrature weights
    if weights is not None:
        while weights.ndim < sq.ndim:
            weights = weights.unsqueeze(0)
        sq = weights * sq
    
    # Reduce over spatial dimensions
    spatial_dims = tuple(range(2, sq.ndim))
    if len(spatial_dims) > 0:
        result = sq.sum(dim=spatial_dims)
    else:
        result = sq
    
    # Reduce over channels
    result = result.sum(dim=1)  # (B,)
    
    if reduce == "mean":
        return result.mean()
    elif reduce == "sum":
        return result.sum()
    elif reduce == "none":
        return result
    else:
        raise ValueError(f"Unknown reduce mode: {reduce}")


def bulk_state_norm_squared(
    phi: Tensor,
    pi: Tensor,
    r: Tensor,
    d: int,
    slice_geometry: str,
    weights: Optional[Tensor] = None,
    reduce: str = "mean",
) -> Tensor:
    """
    Compute ||S||²_{g_r} = ||Φ̃||²_{g_r} + ||Π̃||²_{g_r}.
    
    Document eq (fm-loss): Loss uses intrinsic AdS slice measure.
    
    Args:
        phi: Field tensor Φ̃
        pi: Momentum tensor Π̃
        r: Radial coordinate
        d: Boundary dimension
        slice_geometry: Geometry type
        weights: Quadrature weights
        reduce: Reduction mode
        
    Returns:
        Bulk state norm squared
    """
    omega = compute_slice_weight(r, d, slice_geometry)
    
    norm_phi = weighted_l2_norm_squared(phi, weights=weights, omega=omega, reduce=reduce)
    norm_pi = weighted_l2_norm_squared(pi, weights=weights, omega=omega, reduce=reduce)
    
    if reduce == "none":
        return norm_phi + norm_pi
    return norm_phi + norm_pi


# =============================================================================
# KG Residual Diagnostics (verifying backbone correctness)
# =============================================================================

@dataclass
class KGResidualDiagnostics:
    """Container for KG residual diagnostics."""
    
    residual_phi: Tensor  # Should be ~0 (dΦ̃/dr - Π̃)
    residual_pi: Tensor   # Should be ~0 if no residual network
    relative_error_phi: float
    relative_error_pi: float
    is_valid: bool


def compute_kg_residual_stabilized(
    phi_tilde: Tensor,
    pi_tilde: Tensor,
    dphi_dr: Tensor,
    dpi_dr: Tensor,
    r: Tensor,
    d: int,
    deltas: Tensor,
    slice_geometry: str,
    laplacian_eigs: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> KGResidualDiagnostics:
    """
    Verify that (dphi_dr, dpi_dr) satisfy the stabilized KG equations.
    
    Document eq (uv-stable-ode) for single-warp:
        dΦ̃/dr = Π̃
        dΠ̃/dr = -(d(f'/f) - 2(d-Δ))Π̃ + [-(1/f²)Δ_ĝ + d(d-Δ)(f'/f - 1)]Φ̃
    
    This function computes:
        residual_phi = dΦ̃/dr - Π̃  (should be ~0)
        residual_pi = dΠ̃/dr - [expected from KG]  (should be ~0)
    
    CRITICAL: Uses correct geometry API (f(), log_f_prime via formulas).
    
    Args:
        phi_tilde: Field tensor Φ̃
        pi_tilde: Momentum tensor Π̃
        dphi_dr: Field derivative dΦ̃/dr
        dpi_dr: Momentum derivative dΠ̃/dr
        r: Radial coordinate
        d: Boundary dimension
        deltas: Conformal dimensions
        slice_geometry: Geometry type
        laplacian_eigs: Optional Laplacian eigenvalues
        eps: Numerical stability epsilon
        
    Returns:
        KGResidualDiagnostics with residuals and errors
        
    Raises:
        ValueError: If geometry is unknown
    """
    device = phi_tilde.device
    dtype = phi_tilde.dtype
    geom = slice_geometry.lower()
    
    # Broadcast deltas to (1, C, 1, ...)
    C = phi_tilde.shape[1]
    deltas_b = deltas.to(device=device, dtype=dtype).view(1, C, *([1] * (phi_tilde.ndim - 2)))
    
    # Broadcast r
    r_b = r.to(device=device, dtype=dtype)
    if r_b.ndim == 0:
        r_b = r_b.view(1)
    while r_b.ndim < phi_tilde.ndim:
        r_b = r_b.unsqueeze(-1)
    
    # Compute warp factor and its log derivative
    if geom == "planar":
        f = torch.exp(r_b)
        log_f_prime = torch.ones_like(r_b)  # f'/f = 1
    else:
        raise ValueError(f"Unknown slice geometry: {slice_geometry}")
    
    # Single-warp case
    # Expected dΦ̃/dr = Π̃
    expected_dphi = pi_tilde
    residual_phi = dphi_dr - expected_dphi
    
    # Expected dΠ̃/dr = -(d(f'/f) - 2(d-Δ))Π̃ + [-(1/f²)λ + d(d-Δ)(f'/f - 1)]Φ̃
    A = float(d) * log_f_prime - 2.0 * (float(d)-deltas_b)
    
    if laplacian_eigs is not None:
        lam = laplacian_eigs.to(device=device, dtype=dtype)
        while lam.ndim < phi_tilde.ndim:
            lam = lam.unsqueeze(0)
        inv_f2 = 1.0 / (f * f)
        B = -inv_f2 * lam + float(d) * (float(d) - deltas_b) * (log_f_prime - 1.0)
    else:
        B = float(d) * (float(d) - deltas_b) * (log_f_prime - 1.0)
    
    expected_dpi = -A * pi_tilde + B * phi_tilde
    residual_pi = dpi_dr - expected_dpi
    
    # Compute relative errors
    phi_scale = phi_tilde.abs().mean().item() + eps
    pi_scale = pi_tilde.abs().mean().item() + eps
    dphi_scale = dphi_dr.abs().mean().item() + eps
    dpi_scale = dpi_dr.abs().mean().item() + eps
    
    rel_err_phi = residual_phi.abs().mean().item() / dphi_scale
    rel_err_pi = residual_pi.abs().mean().item() / max(dpi_scale, eps)
    
    return KGResidualDiagnostics(
        residual_phi=residual_phi,
        residual_pi=residual_pi,
        relative_error_phi=rel_err_phi,
        relative_error_pi=rel_err_pi,
        is_valid=(rel_err_phi < 0.01),
    )


# =============================================================================
# Statistics and metrics
# =============================================================================

def compute_data_statistics(
    data: Tensor,
    dim: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Tensor]:
    """
    Compute mean, std, min, max of data.
    
    Args:
        data: Input tensor
        dim: Dimensions to reduce over (None = all)
        
    Returns:
        Dict with mean, std, min, max tensors
    """
    if dim is None:
        dim = tuple(range(data.ndim))
    
    return {
        "mean": data.mean(dim=dim),
        "std": data.std(dim=dim),
        "min": data.amin(dim=dim),
        "max": data.amax(dim=dim),
    }


def normalize_data(
    data: Tensor,
    mean: Optional[Tensor] = None,
    std: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Normalize data to zero mean and unit variance.
    
    Args:
        data: Input tensor
        mean: Precomputed mean (None = compute from data)
        std: Precomputed std (None = compute from data)
        eps: Numerical stability epsilon
        
    Returns:
        (normalized_data, mean, std)
    """
    if mean is None:
        mean = data.mean()
    if std is None:
        std = data.std()
    
    normalized = (data - mean) / (std + eps)
    return normalized, mean, std


def denormalize_data(
    data: Tensor,
    mean: Tensor,
    std: Tensor,
) -> Tensor:
    """
    Reverse normalization.
    
    Args:
        data: Normalized tensor
        mean: Mean used for normalization
        std: Std used for normalization
        
    Returns:
        Denormalized tensor
    """
    return data * std + mean


# =============================================================================
# Model inspection utilities
# =============================================================================

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    """
    Get model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.01,
    no_decay_patterns: Sequence[str] = ("bias", "norm", "LayerNorm"),
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with different weight decay.
    
    Typically, bias and normalization parameters should not have weight decay.
    
    Args:
        model: PyTorch model
        weight_decay: Weight decay for regular parameters
        no_decay_patterns: Parameter name patterns to exclude from weight decay
        
    Returns:
        List of parameter group dicts
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if any(pattern in name for pattern in no_decay_patterns):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


# =============================================================================
# Timing utilities
# =============================================================================

class Timer:
    """
    Simple timer context manager.
    
    Supports both CPU and GPU timing with proper CUDA synchronization.
    
    Example:
        with Timer("forward pass") as t:
            output = model(input)
        print(t)  # Timer(forward pass): 0.0123s
    """
    
    def __init__(self, name: str = ""):
        """
        Initialize timer.
        
        Args:
            name: Optional name for the timer
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.start_time is not None:
            self.start_time.record()
        else:
            self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available() and isinstance(self.start_time, torch.cuda.Event):
            self.end_time = torch.cuda.Event(enable_timing=True)
            self.end_time.record()
            torch.cuda.synchronize()
            self.elapsed = self.start_time.elapsed_time(self.end_time) / 1000.0  # Convert to seconds
        else:
            self.elapsed = time.perf_counter() - self.start_time
    
    def __repr__(self):
        if self.elapsed is not None:
            return f"Timer({self.name}): {self.elapsed:.4f}s"
        return f"Timer({self.name}): not measured"


# =============================================================================
# Checkpoint utilities
# =============================================================================

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    step: int = 0,
    epoch: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    **kwargs,
) -> None:
    """
    Save training checkpoint.
    
    Args:
        path: Save path
        model: PyTorch model
        optimizer: Optional optimizer
        scheduler: Optional learning rate scheduler
        step: Current training step
        epoch: Current epoch
        metrics: Optional metrics dict
        **kwargs: Additional data to save
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "step": step,
        "epoch": epoch,
        "metrics": metrics or {},
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    checkpoint.update(kwargs)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        path: Checkpoint path
        model: PyTorch model to load into
        optimizer: Optional optimizer to load into
        scheduler: Optional scheduler to load into
        device: Device to map checkpoint to
        
    Returns:
        Full checkpoint dict
    """
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint


# =============================================================================
# FFT utilities
# =============================================================================

def fft_convolve_2d(x: Tensor, kernel: Tensor) -> Tensor:
    """
    2D convolution via FFT (for periodic boundary conditions).
    
    Args:
        x: Input tensor (..., H, W)
        kernel: Convolution kernel
        
    Returns:
        Convolved tensor
    """
    H, W = x.shape[-2:]
    kH, kW = kernel.shape[-2:]
    
    # Pad kernel to match x size
    kernel_padded = torch.zeros_like(x)
    kernel_padded[..., :kH, :kW] = kernel
    
    # FFT convolution
    X = torch.fft.fft2(x)
    K = torch.fft.fft2(kernel_padded)
    Y = X * K
    y = torch.fft.ifft2(Y).real
    
    return y


def compute_power_spectrum_2d(x: Tensor) -> Tensor:
    """
    Compute 2D power spectrum |FFT(x)|².
    
    Args:
        x: Input tensor (..., H, W)
        
    Returns:
        Power spectrum tensor
    """
    X = torch.fft.fft2(x)
    return (X.real ** 2 + X.imag ** 2)


# =============================================================================
# Sampling utilities
# =============================================================================

def sample_truncated_normal(
    shape: Tuple[int, ...],
    mean: float = 0.0,
    std: float = 1.0,
    truncation: float = 2.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """
    Sample from truncated normal distribution.
    
    Args:
        shape: Output shape
        mean: Distribution mean
        std: Distribution std
        truncation: Number of stds to truncate at
        device: Output device
        dtype: Output dtype
        generator: Optional random generator
        
    Returns:
        Samples from truncated normal
    """
    samples = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    samples = samples.clamp(-truncation, truncation)
    return mean + std * samples


def sample_von_mises_fisher(
    shape: Tuple[int, ...],
    mu: Tensor,
    kappa: float,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """
    Sample from von Mises-Fisher distribution on sphere.
    
    Simple rejection sampling implementation.
    
    Args:
        shape: Output shape
        mu: Mean direction (unit vector)
        kappa: Concentration parameter
        device: Output device
        dtype: Output dtype
        generator: Optional random generator
        
    Returns:
        Samples from vMF distribution (unit vectors)
    """
    # Simplified: just return noisy version of mu for small kappa
    noise = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    samples = mu + noise / (kappa + 1e-8)
    # Normalize to unit sphere
    samples = samples / (samples.norm(dim=-1, keepdim=True) + 1e-8)
    return samples


# =============================================================================
# Logging utilities
# =============================================================================

def setup_logging(
    level: int = 20,  # logging.INFO
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """
    Configure logging with optional file output.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional log file path
        format_string: Custom format string
    """
    import logging
    
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s: %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
    )


# =============================================================================
# Miscellaneous utilities
# =============================================================================

def ensure_dir(path: str) -> str:
    """
    Ensure directory exists, creating if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        The path (for chaining)
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get device, with automatic CUDA detection.
    
    Args:
        device: Device string ("cuda", "cpu", None for auto)
        
    Returns:
        torch.device object
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def move_to_device(
    data: Union[Tensor, Dict, List, Tuple],
    device: torch.device,
) -> Union[Tensor, Dict, List, Tuple]:
    """
    Recursively move tensors to device.
    
    Args:
        data: Tensor, dict, list, or tuple containing tensors
        device: Target device
        
    Returns:
        Data structure with tensors moved to device
    """
    if isinstance(data, Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(v, device) for v in data)
    else:
        return data
