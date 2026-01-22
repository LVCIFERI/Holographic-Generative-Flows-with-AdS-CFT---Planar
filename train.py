#!/usr/bin/env python3
"""
train.py

Command-line training script for UV-stabilized generative flow matching.

Implementation of Paper Algorithm 1 (Training):
1. Sample data y ~ p_data
2. Sample IR bulk state S_0 ~ μ_IR (Sobolev-weighted Gaussian)
3. Set UV target S_1 = Lift(y) via spectral encoding (paper Section 3.3)
4. Sample t ~ Uniform[0,1]
5. Compute backbone-subtracted loss (paper eq. 35)
6. Update parameters θ

Usage:
    # Basic training with point data (no field encoding)
    python train.py --dataset checkerboard --epochs 100
    
    # With SPECTRAL bulk-boundary propagator encoding (RECOMMENDED)
    python train.py --dataset checkerboard --use_spectral_encoding --epochs 100
    
    # Full example with spectral encoding
    python train.py --dataset gaussian_mixture --use_spectral_encoding \\
        --spectral_n_modes 16 --epochs 500 --lr 3e-4

Paper Section 4 (Experiments):
    The paper uses K=16 spectral modes, L=8 domain width, and 50,000 samples.
    Default values below match the paper's experimental setup.

LAPLACIAN TYPE OPTIONS:
    
    The Laplacian is computed using the diagonal (FFT-based) method:
    
    DIAGONAL (default): Fast diagonal multiplication in spectral space
       - Exact for planar geometry
       - Eigenvalues λ_k = |k|² (paper Section 2.2)
       - Uses FFT for O(N log N) complexity
       
       python train.py --dataset checkerboard --use_spectral_encoding \\
           --slice_geometry planar --laplacian_type diagonal
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# Local imports
from ads_cft.config import ExperimentConfig
from ads_cft.geometry import GeometryConfig, SliceGeometry, make_geometry
from ads_cft.model import (
    FlowModelConfig,
    UVStabilizedFlowMatchingModel,
    PathConfig,
    PathType,
)
from ads_cft.config import IntegratorConfig, ODESolver, DiscretizationConfig
from ads_cft.trainer import Trainer, TrainerConfig, setup_logging
from ads_cft.data_toy import create_sampler, create_toy_dataset, DatasetType
from ads_cft.data_image import (
    is_image_dataset, 
    load_image_dataset, 
    get_image_data_info,
    create_image_dataloaders,
)
from ads_cft.utils import set_seed, count_parameters
from ads_cft.metrics import compute_metrics, compute_bpd_from_model
from ads_cft.visualization import visualize_samples, plot_2d_samples, plot_image_samples, save_clean_data, save_generated_only


# =============================================================================
# Checkerboard Metrics Configuration Helper
# =============================================================================

def get_checkerboard_config(dataset_name: str) -> Optional[Dict]:
    """
    Get checkerboard metrics configuration for a dataset.
    
    Returns configuration dict for checkerboard-style datasets, None otherwise.
    
    The checkerboard metrics (BV, JS_cell, WED, CQS) require knowledge of:
    - domain: bounding box (xmin, xmax, ymin, ymax)
    - cell_size: size of each checkerboard cell
    - parity: which tiles are valid (0 or 1)
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dict with domain, cell_size, parity, lam, mu if checkerboard dataset,
        None otherwise
    """
    dataset_lower = dataset_name.lower()
    
    # Standard 2D checkerboard: 4x4 tiles on [-4, 4]^2
    if dataset_lower == "checkerboard":
        return {
            "domain": (-4.0, 4.0, -4.0, 4.0),
            "cell_size": 2.0,  # 8.0 / 4 tiles = 2.0
            "parity": 0,
            "lam": 1.0,
            "mu": 1.0,
        }
    
    # For other datasets that might have checkerboard-like structure:
    # sphere_checkerboard and hyperboloid_checkerboard are on curved manifolds
    # and would need different coordinate handling (not 2D Euclidean)
    # For now, we only support planar checkerboard
    
    return None


def is_checkerboard_dataset(dataset_name: str) -> bool:
    """Check if dataset is a checkerboard type that supports checkerboard metrics."""
    return get_checkerboard_config(dataset_name) is not None


# =============================================================================
# Note: ExperimentConfig is imported from config.py
# =============================================================================


# =============================================================================
# Configuration I/O
# =============================================================================

def load_config(config_path: Optional[str] = None, **overrides) -> ExperimentConfig:
    """
    Load configuration from YAML/JSON file with CLI overrides.
    
    Args:
        config_path: Path to config file (None for defaults)
        **overrides: CLI overrides
        
    Returns:
        ExperimentConfig instance
    """
    config_dict = {}
    
    if config_path is not None:
        path = Path(config_path)
        if path.suffix in [".yaml", ".yml"]:
            try:
                import yaml
                with open(path) as f:
                    config_dict = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required for YAML configs. Install with: pip install pyyaml")
        elif path.suffix == ".json":
            with open(path) as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unknown config format: {path.suffix}")
    
    # Apply overrides
    config_dict.update({k: v for k, v in overrides.items() if v is not None})
    
    # Handle tuple conversion for deltas
    if "deltas" in config_dict and isinstance(config_dict["deltas"], list):
        config_dict["deltas"] = tuple(config_dict["deltas"])
    
    return ExperimentConfig(**config_dict)


def save_config(config: ExperimentConfig, path: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: ExperimentConfig instance
        path: Output path
    """
    with open(path, "w") as f:
        json.dump(asdict(config), f, indent=2)


# =============================================================================
# Configuration Validation
# =============================================================================

class ConfigurationError(Exception):
    """Raised when configuration parameters are invalid or incompatible."""
    pass


def validate_config(config: ExperimentConfig) -> Tuple[bool, List[str]]:
    """
    Validate configuration parameters for compatibility and physical constraints.
    
    Checks:
    1. Laplacian type compatibility with geometry
    2. Conformal dimension (delta) physical constraints
    3. HSV parameter constraints for different geometries
    4. Symplectic solver requirements
    5. Encoding compatibility
    
    Args:
        config: ExperimentConfig instance to validate
        
    Returns:
        Tuple of (is_valid, list_of_warnings_or_errors)
        
    Raises:
        ConfigurationError: If configuration is invalid and cannot proceed
    """
    errors = []
    warnings = []
    
    # 1. Conformal dimension constraints
    # For AdS/CFT: Δ > d/2 for unitarity bound (d = boundary dimension, typically 2)
    d = 2  # boundary spatial dimension
    min_delta = d / 2  # = 1.0 for d=2
    
    for i, delta in enumerate(config.deltas):
        if delta <= min_delta:
            errors.append(
                f"Conformal dimension Δ_{i} = {delta} violates unitarity bound. "
                f"Required: Δ > d/2 = {min_delta} for d={d}. "
                f"Use --deltas with values > {min_delta}."
            )
        elif delta < min_delta + 0.1:
            warnings.append(
                f"Conformal dimension Δ_{i} = {delta} is very close to unitarity bound. "
                f"This may cause numerical instabilities."
            )
    
    # 3. HSV parameter constraints
    if "hsv" in config.slice_geometry.lower():
        p = config.hsv_p
        
        # General HSV constraint: p ∈ (0, 1]
        if p <= 0 or p > 1:
            errors.append(
                f"HSV parameter p = {p} out of valid range [0, 1). "
                f"Use --hsv_p with 0 < p ≤ 1."
            )
    
    # 2. Symplectic solver requirements
    symplectic_solvers = ["leapfrog", "implicit_midpoint"]
    if config.ode_solver in symplectic_solvers:
        if config.residual_type != "potential":
            warnings.append(
                f"Symplectic solver '{config.ode_solver}' works best with potential residual. "
                f"Got: residual_type='{config.residual_type}'. "
                f"Consider using --residual_type potential for energy conservation."
            )
    
    # 3. Path type and backbone compatibility
    if config.path_type == "linear" and getattr(config, 'backbone_scale', 1.0) > 0:
        warnings.append(
            "Linear path type with KG backbone may not benefit from backbone structure. "
            "Consider using --path_type hermite for full backbone utilization."
        )
    
    # 4. Image encoding requirements
    if config.use_image_encoding:
        image_datasets = ["mnist", "fashion_mnist", "cifar10", "cifar100"]
        if not any(ds in config.dataset.lower() for ds in image_datasets):
            warnings.append(
                f"Image encoding enabled but dataset '{config.dataset}' may not be an image dataset."
            )
    
    is_valid = len(errors) == 0
    messages = errors + warnings
    
    return is_valid, messages


def validate_and_report(config: ExperimentConfig, strict: bool = True) -> None:
    """
    Validate configuration and report issues.
    
    Args:
        config: ExperimentConfig instance
        strict: If True, raise exception on errors. If False, only warn.
        
    Raises:
        ConfigurationError: If strict=True and validation fails
    """
    is_valid, messages = validate_config(config)
    
    if not messages:
        return
    
    # Separate errors and warnings
    errors = [m for m in messages if not m.startswith("Warning") and "may" not in m.lower() and "consider" not in m.lower()]
    warnings = [m for m in messages if m not in errors]
    
    # Report warnings
    for warning in warnings:
        logging.warning(f"Configuration warning: {warning}")
    
    # Report errors
    if errors:
        error_msg = "\n".join([f"  - {e}" for e in errors])
        full_msg = f"Configuration validation failed:\n{error_msg}"
        
        if strict:
            raise ConfigurationError(full_msg)
        else:
            logging.error(full_msg)


# =============================================================================
# Experiment Naming
# =============================================================================

def build_experiment_name(config: ExperimentConfig, include_timestamp: bool = False) -> str:
    """
    Build a comprehensive experiment name from configuration.
    
    Naming convention:
        {method}_{dataset}_{path}_{geometry}[_{hsv}][_{sigma}][_{radii}][_{solver}][_{residual}][_{laplacian}][_{laplacian_method}][_{delta}][_{encoding}][_{timestamp}]
    
    Method prefixes:
        - vanilla: Vanilla CNN flow matching (--use_vanilla_cnn)
        - nokg: AdS encoding but no KG backbone (backbone_scale=0)
        - ads: Full AdS model with KG backbone
    
    Examples:
        vanilla_mnist_linear_planar
        nokg_mnist_linear_planar_image
        ads_mnist_hermite_planar_image
        ads_checkerboard_hermite_planar
    
    Args:
        config: ExperimentConfig instance
        include_timestamp: Whether to append timestamp for uniqueness
        
    Returns:
        Descriptive experiment name string
    """
    parts = []
    
    # 1. Method prefix - detect model type
    if getattr(config, 'use_vanilla_cnn', False):
        parts.append("vanilla")
    elif config.backbone_scale == 0.0:
        parts.append("nokg")
    else:
        parts.append("ads")
    
    # 2. Dataset
    parts.append(config.dataset)
    
    # 3. Path type
    parts.append(config.path_type)
    
    # 4. Geometry (with HSV parameter if applicable)
    geom = config.slice_geometry
    if "hsv" in geom.lower():
        # Include HSV p value
        p_str = f"p{config.hsv_p:.2f}".rstrip('0').rstrip('.')
        parts.append(f"{geom}_{p_str}")
        
        # Include u bounds if using them (non-default)
        if config.hsv_use_u_bounds:
            u_uv = config.hsv_u_uv
            u_ir = config.hsv_u_ir
            # Only include if non-default (0.1, 1.0)
            if abs(u_uv - 0.1) > 0.01 or abs(u_ir - 1.0) > 0.01:
                u_str = f"u{u_uv:.1f}-{u_ir:.1f}".replace('.', '')
                parts.append(u_str)
    else:
        parts.append(geom)
    
    # 5. Radial bounds (only if non-default: 0.0, 1.0)
    default_r_ir, default_r_uv = 0.0, 1.0
    if abs(config.r_ir - default_r_ir) > 0.001 or abs(config.r_uv - default_r_uv) > 0.001:
        # Only include for non-HSV with u bounds (HSV computes r from u)
        if not ("hsv" in geom.lower() and config.hsv_use_u_bounds):
            r_str = f"r{config.r_ir:.2f}-{config.r_uv:.2f}".replace('.', '')
            parts.append(r_str)
    
    # 6. ODE Solver (only if non-default)
    solver = config.ode_solver
    if solver != "rk4":
        parts.append(solver)
    
    # 7. Residual type (only if non-default)
    if config.residual_type != "direct":
        parts.append(config.residual_type)
    
    # 8. Laplacian type (only if non-default)
    laplacian = config.laplacian_type
    # Laplacian type is always diagonal, no need to add to name
    
    # 9. Conformal dimension (only if non-default)
    # Default is (1.5, 1.5) - include if different
    default_deltas = (1.5, 1.5)
    if config.deltas != default_deltas:
        # Use first delta value (or unique values if heterogeneous)
        unique_deltas = set(config.deltas)
        if len(unique_deltas) == 1:
            d_str = f"d{config.deltas[0]:.1f}".rstrip('0').rstrip('.')
        else:
            d_str = "d" + "_".join(f"{d:.1f}".rstrip('0').rstrip('.') for d in config.deltas)
        parts.append(d_str)
    
    # 10. Path variant (radial Hermite)
    # Only include if explicitly enabled for non-HSV (HSV auto-enables it)
    if config.use_radial_hermite and "hsv" not in geom.lower():
        parts.append("radial")
    
    # 11. Loss weighting (only if disabled, since enabled is default)
    if not config.use_omega_weighting:
        parts.append("noweight")
    
    # 12. Encoding type (if using special encoding)
    if config.use_spectral_encoding:
        parts.append(f"spectral{config.spectral_n_modes}")
    elif config.use_image_encoding:
        parts.append("image")
    elif config.use_holographic_encoding:
        parts.append("holo")
    elif config.use_holographic_grid:
        parts.append("hologrid")
    elif config.use_field_encoding:
        parts.append("field")
    
    # 13. Timestamp (optional, for unique run directories)
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.append(timestamp)
    
    return "_".join(parts)


def parse_experiment_name(exp_name: str) -> Dict[str, Any]:
    """
    Parse experiment name back into components.
    
    This is the inverse of build_experiment_name(), used by evaluate.py
    to understand experiment configurations from folder names.
    
    Args:
        exp_name: Experiment name string
        
    Returns:
        Dictionary with parsed components:
            - method: str (e.g., "ads")
            - dataset: str (e.g., "checkerboard")
            - path_type: str (e.g., "hermite")
            - geometry: str (e.g., "planar", "planar_hsv")
            - hsv_p: float or None
            - hsv_u_bounds: tuple or None (u_uv, u_ir)
            - r_bounds: tuple or None (r_ir, r_uv)
            - use_radial_hermite: bool
            - use_omega_weighting: bool
            - ode_solver: str (e.g., "rk4")
            - residual_type: str (e.g., "direct")
            - laplacian_type: str (e.g., "diagonal")
            - deltas: tuple or None
            - encoding: str or None
            - spectral_modes: int or None
            - timestamp: str or None
    """
    result = {
        "method": "ads",
        "dataset": "unknown",
        "path_type": "hermite",
        "geometry": "planar",
        "hsv_p": None,
        "hsv_u_bounds": None,
        "r_bounds": None,
        "use_radial_hermite": False,
        "use_omega_weighting": True,
        "ode_solver": "rk4",
        "residual_type": "direct",
        "laplacian_type": "diagonal",
        "deltas": None,
        "encoding": None,
        "spectral_modes": None,
        "timestamp": None,
    }
    
    parts = exp_name.split("_")
    
    if len(parts) < 4:
        # Minimal parsing for legacy names
        if len(parts) >= 1:
            result["method"] = parts[0]
        if len(parts) >= 2:
            result["dataset"] = parts[1]
        if len(parts) >= 3:
            result["path_type"] = parts[2]
        return result
    
    # Standard parsing
    idx = 0
    
    # 1. Method
    result["method"] = parts[idx]
    idx += 1
    
    # 2. Dataset
    result["dataset"] = parts[idx]
    idx += 1
    
    # 3. Path type
    result["path_type"] = parts[idx]
    idx += 1
    
    # 4. Geometry (may include HSV p value)
    geom_part = parts[idx]
    idx += 1
    
    # Check for HSV p value in next part
    if idx < len(parts) and parts[idx].startswith("p") and not parts[idx].startswith("potential"):
        try:
            p_str = parts[idx][1:]  # Remove 'p' prefix
            result["hsv_p"] = float(p_str)
            result["geometry"] = geom_part
            idx += 1
        except ValueError:
            result["geometry"] = geom_part
    else:
        result["geometry"] = geom_part
    
    # Parse remaining optional parts
    while idx < len(parts):
        part = parts[idx]
        
        # Check for u bounds (u01-10 format)
        if part.startswith("u") and "-" in part and not part.startswith("use"):
            try:
                u_str = part[1:]  # Remove 'u' prefix
                if "-" in u_str:
                    u_parts = u_str.split("-")
                    # Convert back: 01 -> 0.1, 10 -> 1.0
                    u_uv = float(u_parts[0]) / 10.0 if len(u_parts[0]) <= 2 else float(u_parts[0])
                    u_ir = float(u_parts[1]) / 10.0 if len(u_parts[1]) <= 2 else float(u_parts[1])
                    result["hsv_u_bounds"] = (u_uv, u_ir)
            except (ValueError, IndexError):
                pass
        # Check for radii bounds (r010-200 format)
        elif part.startswith("r") and "-" in part and not part.startswith("rk"):
            try:
                r_str = part[1:]  # Remove 'r' prefix
                r_parts = r_str.split("-")
                # Convert back: 010 -> 0.10, 200 -> 2.00
                r_ir = float(r_parts[0]) / 100.0 if len(r_parts[0]) >= 3 else float(r_parts[0])
                r_uv = float(r_parts[1]) / 100.0 if len(r_parts[1]) >= 3 else float(r_parts[1])
                result["r_bounds"] = (r_ir, r_uv)
            except (ValueError, IndexError):
                pass
        # Check for ODE solvers
        elif part in ["euler", "heun", "midpoint", "rk4", "leapfrog", "implicit_midpoint"]:
            result["ode_solver"] = part
        # Check for residual types
        elif part in ["direct", "potential"]:
            result["residual_type"] = part
        # Check for Laplacian types
        elif part == "diagonal":
            result["laplacian_type"] = part
        # Check for delta values (d1.5, d2.0, etc.)
        elif part.startswith("d") and not part.startswith("direct") and not part.startswith("diagonal"):
            try:
                delta_str = part[1:]  # Remove 'd' prefix
                # Handle single delta or multiple (d1.5 or d1.5_2.5 encoded differently)
                result["deltas"] = (float(delta_str),)
            except ValueError:
                pass  # Not a delta value
        # Check for radial Hermite flag
        elif part == "radial":
            result["use_radial_hermite"] = True
        # Check for no omega weighting flag
        elif part == "noweight":
            result["use_omega_weighting"] = False
        # Check for encoding types
        elif part.startswith("spectral"):
            result["encoding"] = "spectral"
            # Extract mode count if present
            mode_str = part[8:]  # After "spectral"
            if mode_str.isdigit():
                result["spectral_modes"] = int(mode_str)
        elif part in ["image", "holo", "hologrid", "field"]:
            result["encoding"] = part
        # Check for timestamp (YYYYMMDD_HHMMSS format)
        elif len(part) == 15 and part[8:9] == "_":
            result["timestamp"] = part
        elif len(part) == 8 and part.isdigit():
            # Just date part
            if idx + 1 < len(parts) and len(parts[idx + 1]) == 6 and parts[idx + 1].isdigit():
                result["timestamp"] = f"{part}_{parts[idx + 1]}"
                idx += 1
        
        idx += 1
    
    return result


def get_experiment_signature(config: ExperimentConfig) -> str:
    """
    Get a short signature for quick identification.
    
    Used for logging and progress displays.
    
    Args:
        config: ExperimentConfig instance
        
    Returns:
        Short signature string like "checkerboard/hermite/hsv_p0.5/leapfrog"
    """
    parts = [config.dataset, config.path_type]
    
    geom = config.slice_geometry
    if "hsv" in geom.lower():
        parts.append(f"{geom}_p{config.hsv_p:.2f}")
    else:
        parts.append(geom)
    
    if config.ode_solver != "rk4":
        parts.append(config.ode_solver)
    
    if config.residual_type != "direct":
        parts.append(config.residual_type)
    
    return "/".join(parts)


# =============================================================================
# Model creation
# =============================================================================

def create_model(
    config: ExperimentConfig,
    data_shape: Tuple[int, ...],
) -> UVStabilizedFlowMatchingModel:
    """
    Create UV-stabilized flow matching model from config.
    
    Document references:
    - Section 2: Geometry setup
    - Section 3-4: Backbone and residual
    - Section 5: Lift and readout
    - Section 7: IR prior
    
    Encoding modes:
    - No encoding: n_channels must equal data_dim
    - Field encoding: n_channels can be independent (fields on grid)
    
    Args:
        config: Experiment configuration
        data_shape: Shape of data samples
        
    Returns:
        UVStabilizedFlowMatchingModel instance
    """
    print("[DEBUG] create_model() started", flush=True)
    
    # Compute data dimension and handle image vs point data
    if config.use_image_encoding and len(data_shape) == 3:
        # Image data: shape is (C, H, W)
        n_image_channels = data_shape[0]
        image_height = data_shape[1]
        image_width = data_shape[2]
        
        # For images, d = 2 always (2D spatial boundary)
        d = 2
        data_dim = n_image_channels  # For deltas matching
        
        print(f"[IMAGE] Image shape: {n_image_channels}×{image_height}×{image_width}", flush=True)
        print(f"[IMAGE] Boundary dimension d = 2 (2D image plane)", flush=True)
        print(f"[IMAGE] Number of field channels: {n_image_channels}", flush=True)
    else:
        # Point data: compute flat dimension
        data_dim = 1
        for s in data_shape:
            data_dim *= s
        
        # Auto-discover d (boundary dimension) from data_dim if not specified
        if config.d is None:
            d = data_dim
            print(f"[AUTO] Boundary dimension d = {d} (from data shape {data_shape})", flush=True)
            print(f"[AUTO] AdS bulk dimension = {d + 1}", flush=True)
        else:
            d = config.d
            if d != data_dim:
                print(f"[INFO] config.d={d} differs from data_dim={data_dim}", flush=True)
    
    print(f"[DEBUG] d={d}, data_dim={data_dim}", flush=True)
    
    # Auto-expand deltas for point data if needed
    deltas = config.deltas
    n_deltas = len(deltas)
    
    # Determine if we're using any field encoding
    using_grid_encoding = config.use_field_encoding or config.use_holographic_grid
    
    if config.use_image_encoding:
        # For images, deltas must match number of image channels
        n_image_channels = data_shape[0] if len(data_shape) == 3 else 1
        print(f"Image encoding (DIRECT FFT): {n_image_channels} image channels", flush=True)
        print(f"  Images ARE boundary fields - no propagator needed", flush=True)
        print(f"  Laplacian: DIAGONAL (super fast!)", flush=True)
        if n_deltas < n_image_channels:
            base_delta = deltas[0]
            deltas = tuple([base_delta] * n_image_channels)
            print(f"  Auto-expanded deltas to {n_image_channels} channels: {deltas}", flush=True)
        elif n_deltas > n_image_channels:
            deltas = deltas[:n_image_channels]
            print(f"  Truncated deltas to {n_image_channels} channels: {deltas}", flush=True)
    elif config.use_spectral_encoding:
        print(f"Spectral encoding (RECOMMENDED): {n_deltas} channels, {config.spectral_n_modes} modes")
        print(f"  Fast: O(K²) memory, diagonal Laplacian")
        print(f"  Faithful: Exact AdS/CFT in Fourier space")
        # For spectral, we need deltas to match data_dim for point decoding
        if n_deltas < data_dim:
            base_delta = deltas[0]
            deltas = tuple([base_delta] * data_dim)
            print(f"  Auto-expanded deltas to {data_dim} channels: {deltas}")
        elif n_deltas > data_dim:
            deltas = deltas[:data_dim]
            print(f"  Truncated deltas to {data_dim} channels: {deltas}")
    elif config.use_holographic_encoding:
        print(f"Holographic encoding (GRID-FREE): {n_deltas} channels")
        print(f"  Memory efficient: O(C) per sample")
        # Grid-free needs n_channels = data_dim (same as no-encoding)
        if n_deltas < data_dim:
            base_delta = deltas[0]
            deltas = tuple([base_delta] * data_dim)
            print(f"  Auto-expanded deltas to {data_dim} channels: {deltas}")
        elif n_deltas > data_dim:
            deltas = deltas[:data_dim]
            print(f"  Truncated deltas to {data_dim} channels: {deltas}")
    elif config.use_holographic_grid:
        print(f"Holographic encoding (GRID-BASED): {n_deltas} channels, grid {config.field_grid_size}")
        print(f"  WARNING: Memory heavy - O(H×W) per sample!")
    elif using_grid_encoding:
        print(f"Gaussian field encoding: {n_deltas} channels, grid {config.field_grid_size}")
    else:
        # Original behavior: match deltas to data dimension
        if n_deltas < data_dim:
            base_delta = deltas[0]
            deltas = tuple([base_delta] * data_dim)
            print(f"Auto-expanded deltas from {n_deltas} to {data_dim} channels: {deltas}")
        elif n_deltas > data_dim:
            deltas = deltas[:data_dim]
            print(f"Truncated deltas to {data_dim} channels: {deltas}")
    
    # Geometry config
    geom_cfg = GeometryConfig(
        d=d,  # Use auto-discovered d
        slice_geometry=SliceGeometry(config.slice_geometry),
        dtype=torch.float32,
        device=config.device,
        hsv_p=config.hsv_p,
        hsv_ads_threshold=config.hsv_ads_threshold,
    )
    
    # SAFETY CHECK: For HSV geometries with p > 0, r_IR = 0 causes singularities
    r_ir = config.r_ir
    r_uv = config.r_uv
    hsv_geometries = ["planar_hsv"]
    if config.slice_geometry in hsv_geometries and config.hsv_p > 0:
        # =================================================================
        # HSV U-BOUNDS: Stable training for ALL p ∈ (0, 1]
        # =================================================================
        # The HSV coordinate u = (pr)^{1/p} is natural:
        #   - Appears directly in propagator K̂(u,k) = (|k|u)^β K_β(|k|u)
        #   - p = 1: u = r (flat), p → 0: u = e^{-r} = z (AdS)
        #   - r = u^p / p, so r > 0 always when u > 0
        #
        # This ensures r_ir stays safely away from singularity for ALL p.
        # =================================================================
        use_u_bounds = (
            config.hsv_use_u_bounds 
            and config.hsv_p > config.hsv_ads_threshold
        )
        
        if use_u_bounds:
            # Create temporary geometry to compute r bounds
            temp_geom = make_geometry(geom_cfg)
            r_ir, r_uv = temp_geom.get_r_bounds_from_u(
                config.hsv_u_uv, config.hsv_u_ir
            )
            print(f"[HSV] Using u-bounds: u ∈ [{config.hsv_u_uv}, {config.hsv_u_ir}]")
            print(f"[HSV] Computed r bounds: r_ir={r_ir:.6f}, r_uv={r_uv:.6f}")
            print(f"[HSV] Note: r_uv < r_ir for HSV (small r = UV boundary, large r = IR bulk)")
            print(f"[HSV] Bessel argument range: |k|u ∈ [|k|×{config.hsv_u_uv}, |k|×{config.hsv_u_ir}]")
        else:
            # Legacy behavior or AdS limit: use fixed r bounds with safety check
            if config.hsv_p <= config.hsv_ads_threshold:
                print(f"[HSV] p={config.hsv_p} <= threshold={config.hsv_ads_threshold}, using AdS formulas")
            min_safe_r_ir = 0.1  # Safe minimum to avoid r=0 singularity
            if r_ir < min_safe_r_ir:
                print(f"[WARNING] {config.slice_geometry} geometry (p={config.hsv_p}) has singularity at r=0")
                print(f"[WARNING] Auto-adjusting r_IR from {r_ir} to {min_safe_r_ir}")
                r_ir = min_safe_r_ir
    
    # Discretization config (point data for toy datasets when not using field encoding)
    disc_cfg = DiscretizationConfig(
        representation="spectral",
        grid_shape=None,  # Point data (overridden if field encoding is used)
    )
    
    # Path config
    path_cfg = PathConfig(
        path_type=PathType(config.path_type),
    )
    
    # Integrator config (including symplectic options)
    int_cfg = IntegratorConfig(
        solver=ODESolver(config.ode_solver),
        n_steps=config.ode_n_steps,
        use_affine_stepping=config.use_affine_stepping,
        implicit_tol=config.implicit_tol,
        implicit_max_iter=config.implicit_max_iter,
    )
    
    # Residual type
    from ads_cft.config import ResidualType
    residual_type = ResidualType(config.residual_type)
    
    # Model config
    model_cfg = FlowModelConfig(
        d=d,  # Use auto-discovered d
        deltas=deltas,
        discretization=disc_cfg,
        geometry=geom_cfg,
        r_ir=r_ir,  # Use adjusted value (safe for HSV geometries)
        r_uv=r_uv,
        hsv_propagator_mode=config.hsv_propagator_mode if config.hsv_propagator_mode else "bessel",
        lift_noise_sigma=config.lift_noise_sigma,
        prior_c_phi=config.prior_c_phi,
        prior_s_phi=config.prior_s_phi,
        prior_c_pi=config.prior_c_pi,
        prior_s_pi=config.prior_s_pi,
        hermite_pi_scale=config.hermite_pi_scale,
        backbone_scale=config.backbone_scale,
        residual_hidden=config.residual_hidden,
        residual_depth=config.residual_depth,
        residual_time_emb_dim=config.residual_time_emb_dim,
        residual_type=residual_type,  # DIRECT or POTENTIAL (symplectic)
        cnn_hidden=config.cnn_hidden,
        cnn_depth=config.cnn_depth,
        mlp_hidden=config.mlp_hidden,
        mlp_depth=config.mlp_depth,
        path_config=path_cfg,
        use_radial_hermite=config.use_radial_hermite,
        use_omega_weighting=config.use_omega_weighting,
        integrator=int_cfg,
        # Legacy Gaussian field encoding
        use_field_encoding=config.use_field_encoding,
        # Spectral encoding (RECOMMENDED)
        use_spectral_encoding=config.use_spectral_encoding,
        spectral_n_modes=config.spectral_n_modes,
        spectral_decode_resolution=config.spectral_decode_resolution,
        spectral_decode_method=config.spectral_decode_method,
        spectral_random_feature_scale=config.spectral_random_feature_scale,
        # Holographic encoding options
        use_holographic_encoding=config.use_holographic_encoding,  # Grid-free
        use_holographic_grid=config.use_holographic_grid,  # Grid-based
        # Shared field parameters
        field_grid_size=config.field_grid_size,
        field_domain=config.field_domain,
        field_sigma=config.field_sigma,
        field_temperature=config.field_temperature,
        # Holographic-specific parameters
        holographic_regularization=config.holographic_regularization,
        holographic_normalize=config.holographic_normalize,
        # Image encoding
        use_image_encoding=config.use_image_encoding,
        image_shape=data_shape if config.use_image_encoding and len(data_shape) == 3 else None,
    )
    
    print("[DEBUG] ModelConfig created, now creating UVStabilizedFlowMatchingModel...", flush=True)
    
    # Create model
    model = UVStabilizedFlowMatchingModel(
        config=model_cfg,
        data_shape=data_shape,
        device=torch.device(config.device),
    )
    
    print("[DEBUG] UVStabilizedFlowMatchingModel created successfully!", flush=True)
    return model


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(
    model: UVStabilizedFlowMatchingModel,
    epoch: int,
    real_data: Optional[torch.Tensor] = None,
    n_gen: int = 64,
    n_real: int = 64,
    save_dir: Optional[str] = None,
    checkerboard_config: Optional[Dict] = None,
    model_type: str = "AdS",
    compute_bpd: bool = False,
    bpd_n_samples: int = 100,
    bpd_n_hutchinson: int = 1,
) -> Dict[str, float]:
    """
    Evaluate model by generating samples.
    
    Document Algorithm 2: Sample from IR prior, integrate, readout.
    
    Args:
        model: Model to evaluate
        epoch: Current epoch (for labeling)
        real_data: Real data for comparison
        n_gen: Number of samples to generate
        n_real: Number of real samples to use
        save_dir: Directory to save visualizations
        checkerboard_config: If provided, compute checkerboard-specific metrics (BV, JS_cell, WED, CQS)
        model_type: Model type for visualization title (e.g., "AdS", "NoKG", "Vanilla")
        compute_bpd: If True, compute bits per dimension (exact log-likelihood)
        bpd_n_samples: Number of samples for BPD computation
        bpd_n_hutchinson: Number of Hutchinson samples for trace estimation
        
    Returns:
        Dict of evaluation metrics
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Use a fresh random generator based on current time to ensure different samples
    # This prevents identical outputs even if global RNG state happens to be the same
    import time as time_module
    viz_seed = int(time_module.time() * 1000) % (2**31)
    torch.manual_seed(viz_seed)
    
    # Debug: print model info
    print(f"[EVAL] Model type: {model_type}", flush=True)
    print(f"[EVAL] Encoding type: {model.encoding_type}", flush=True)
    print(f"[EVAL] Viz seed: {viz_seed}", flush=True)
    if hasattr(model, 'residual'):
        n_params = sum(p.numel() for p in model.residual.parameters())
        print(f"[EVAL] Residual params: {n_params:,}", flush=True)
    
    # Generate samples (no gradients needed)
    with torch.no_grad():
        start_time = time.time()
        samples = model.sample(n_gen)
        gen_time = time.time() - start_time
    
    print(f"[EVAL] Samples shape: {samples.shape}", flush=True)
    print(f"[EVAL] Samples mean: {samples.mean():.4f}, std: {samples.std():.4f}", flush=True)
    
    # Compute basic statistics
    samples_np = samples.cpu()
    
    metrics = {
        "gen_time": gen_time,
        "samples_mean": float(samples_np.mean()),
        "samples_std": float(samples_np.std()),
        "samples_min": float(samples_np.min()),
        "samples_max": float(samples_np.max()),
    }
    
    # Check if this is image data (3D or 4D: B, C, H, W or B, H, W)
    is_image = len(samples.shape) >= 3
    
    if save_dir is not None:
        save_path = Path(save_dir) / f"samples_epoch_{epoch:04d}.png"
        
        if is_image:
            plot_image_samples(
                real_data[:n_real] if real_data is not None else None,
                samples,
                title=f"Epoch {epoch} - {model_type}",
                save_path=str(save_path),
            )
            
            # Save generated-only image grid
            save_generated_only(
                samples,
                save_path=str(Path(save_dir) / f"generated_epoch_{epoch:04d}.png"),
                title=f"Generated - {model_type} (Epoch {epoch})",
            )
        elif samples.shape[-1] == 2:
            plot_2d_samples(
                real_data[:n_real] if real_data is not None else None,
                samples,
                title=f"Epoch {epoch} - {model_type}",
                save_path=str(save_path),
            )
        
        metrics["sample_plot_path"] = str(save_path)
    
    # Compute distribution metrics (no gradients needed)
    if real_data is not None:
        try:
            with torch.no_grad():
                real_subset = real_data[:n_real]
                gen_subset = samples[:n_gen]
                
                # Ensure same device
                real_subset = real_subset.to(gen_subset.device)
                
                eval_metrics = compute_metrics(
                    real_subset, gen_subset, 
                    is_image=is_image, 
                    checkerboard_config=checkerboard_config
                )
                metrics.update(eval_metrics)
            
            # Save metrics JSON if save_dir is provided
            if save_dir is not None:
                metrics_path = Path(save_dir) / f"metrics_epoch_{epoch:04d}.json"
                # Convert to serializable types
                serializable_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, (torch.Tensor,)):
                        serializable_metrics[k] = float(v.cpu().numpy())
                    elif hasattr(v, 'item'):
                        serializable_metrics[k] = v.item()
                    else:
                        serializable_metrics[k] = v
                serializable_metrics["epoch"] = epoch
                # Set method based on model_type
                if "Vanilla" in model_type:
                    serializable_metrics["method"] = "vanilla_cnn_flow_matching"
                elif "No KG" in model_type:
                    serializable_metrics["method"] = "nokg_flow_matching"
                else:
                    serializable_metrics["method"] = "ads_flow_matching"
                
                with open(metrics_path, 'w') as f:
                    json.dump(serializable_metrics, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Metrics computation failed: {e}")
    
    # Compute BPD (bits per dimension) if requested
    # NOTE: BPD computation needs gradients for Hutchinson trace estimator, 
    # so it's outside torch.no_grad() context
    if compute_bpd and real_data is not None:
        try:
            print(f"[EVAL] Computing BPD with {bpd_n_samples} samples, {bpd_n_hutchinson} Hutchinson samples...")
            
            # Create a simple dataloader from real_data
            from torch.utils.data import TensorDataset, DataLoader
            bpd_data = real_data[:bpd_n_samples].to(device)
            bpd_dataset = TensorDataset(bpd_data)
            bpd_loader = DataLoader(bpd_dataset, batch_size=min(32, bpd_n_samples), shuffle=False)
            
            bpd_metrics = compute_bpd_from_model(
                model=model,
                data_loader=bpd_loader,
                n_hutchinson_samples=bpd_n_hutchinson,
                device=device,
            )
            
            metrics["bpd"] = bpd_metrics["bpd"]
            metrics["bpd_std"] = bpd_metrics["bpd_std"]
            metrics["log_likelihood"] = bpd_metrics["log_likelihood"]
            metrics["log_prior"] = bpd_metrics["log_prior"]
            metrics["log_det_jacobian"] = bpd_metrics["log_det_jacobian"]
            metrics["bpd_n_samples"] = bpd_metrics["n_samples"]
            
            print(f"[EVAL] BPD: {bpd_metrics['bpd']:.4f} ± {bpd_metrics['bpd_std']:.4f}")
            print(f"[EVAL] Log-likelihood: {bpd_metrics['log_likelihood']:.4f}")
            
            # Update JSON file with BPD metrics
            if save_dir is not None:
                metrics_path = Path(save_dir) / f"metrics_epoch_{epoch:04d}.json"
                # Convert to serializable types
                serializable_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, (torch.Tensor,)):
                        serializable_metrics[k] = float(v.cpu().numpy())
                    elif hasattr(v, 'item'):
                        serializable_metrics[k] = v.item()
                    else:
                        serializable_metrics[k] = v
                serializable_metrics["epoch"] = epoch
                # Set method based on model_type
                if "Vanilla" in model_type:
                    serializable_metrics["method"] = "vanilla_cnn_flow_matching"
                elif "No KG" in model_type:
                    serializable_metrics["method"] = "nokg_flow_matching"
                else:
                    serializable_metrics["method"] = "ads_flow_matching"
                
                with open(metrics_path, 'w') as f:
                    json.dump(serializable_metrics, f, indent=2)
            
        except Exception as e:
            print(f"[WARNING] BPD computation failed: {e}")
            import traceback
            traceback.print_exc()
    
    return metrics


# =============================================================================
# Main training function
# =============================================================================

def train(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Main training function implementing Document Algorithm 1.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dict of training results
    """
    # Setup
    set_seed(config.seed)
    setup_logging(logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Validate configuration before proceeding
    logger.info("Validating configuration...")
    validate_and_report(config, strict=True)
    logger.info("Configuration validated successfully.")
    
    # Create output directory with comprehensive experiment name
    run_name = build_experiment_name(config, include_timestamp=True)
    output_dir = Path(config.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    
    # Save config
    save_config(config, str(output_dir / "config.json"))
    
    # Log experiment signature
    exp_signature = get_experiment_signature(config)
    logger.info(f"Experiment: {config.name}")
    logger.info(f"Signature: {exp_signature}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {config.device}")
    
    # Log configuration
    logger.info(f"Path type: {config.path_type}")
    logger.info(f"Backbone scale: {config.backbone_scale}")
    logger.info(f"Hermite pi scale: {config.hermite_pi_scale}")
    
    # Log encoding mode
    if config.use_image_encoding:
        logger.info("Encoding: IMAGE (direct FFT for image data)")
    elif config.use_spectral_encoding:
        logger.info(f"Encoding: SPECTRAL (K={config.spectral_n_modes} modes, diagonal Laplacian)")
    elif config.use_holographic_encoding:
        logger.info("Encoding: HOLOGRAPHIC GRID-FREE (memory efficient)")
    elif config.use_holographic_grid:
        logger.info("Encoding: HOLOGRAPHIC GRID-BASED (memory heavy)")
    elif config.use_field_encoding:
        logger.info("Encoding: GAUSSIAN (legacy Gaussian blobs)")
    else:
        logger.info("Encoding: NONE (raw point data)")
    
    # Create datasets
    logger.info(f"Creating dataset: {config.dataset}")
    
    # Check if this is an image dataset
    if is_image_dataset(config.dataset):
        logger.info(f"[IMAGE] Loading image dataset: {config.dataset}")
        
        # Get dataset info
        img_info = get_image_data_info(config.dataset)
        logger.info(f"[IMAGE] Channels: {img_info['n_channels']}, Size: {img_info['height']}×{img_info['width']}")
        logger.info(f"[IMAGE] Classes: {img_info['n_classes']}, Boundary dim: d={img_info['d']}")
        
        # Auto-enable image encoding if not already set
        if not config.use_image_encoding:
            logger.info("[IMAGE] Auto-enabling image encoding (use_image_encoding=True)")
            config = ExperimentConfig(
                **{**asdict(config), "use_image_encoding": True}
            )
        
        # Use smaller viz batch for images (64 for 8x8 grid)
        if config.n_viz_real > 64:
            config = ExperimentConfig(
                **{**asdict(config), "n_viz_real": 64, "n_viz_gen": 64}
            )
        
        # Load image datasets
        train_dataset = load_image_dataset(
            config.dataset,
            root="./data",
            train=True,
            download=True,
            normalize_range=(-1.0, 1.0),
            n_samples=config.n_train_samples,
        )
        val_dataset = load_image_dataset(
            config.dataset,
            root="./data",
            train=False,
            download=True,
            normalize_range=(-1.0, 1.0),
            n_samples=config.n_val_samples,
        )
        
        data_shape = train_dataset.get_data_shape()
        logger.info(f"[IMAGE] Data shape: {data_shape}")
        
        # Compute data statistics from a batch
        batch_size = min(1000, len(train_dataset))
        sample_data = torch.stack([train_dataset[i][0] for i in range(batch_size)])
        data_mean = sample_data.mean()
        data_std = sample_data.std()
        logger.info(f"[IMAGE] Data mean: {data_mean:.4f}, std: {data_std:.4f}")
        
    else:
        # Standard toy dataset
        dataset_type = DatasetType(config.dataset)
        
        train_dataset = create_toy_dataset(
            dataset_type=dataset_type,
            n_samples=config.n_train_samples,
            seed=config.seed,
        )
        
        val_dataset = create_toy_dataset(
            dataset_type=dataset_type,
            n_samples=config.n_val_samples,
            seed=config.seed + 1,
        )
        
        data_shape = train_dataset.data_shape
        logger.info(f"Data shape: {data_shape}")
        
        # Compute data statistics for normalization
        data_mean = train_dataset.data.mean()
        data_std = train_dataset.data.std()
        logger.info(f"Data mean: {data_mean:.4f}, std: {data_std:.4f}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model(config, data_shape)
    
    # Set normalization - must use register_buffer to save in state_dict
    model.register_buffer("data_mean", data_mean.clone().detach())
    model.register_buffer("data_std", data_std.clone().detach())
    
    n_params = count_parameters(model)
    logger.info(f"Model parameters: {n_params:,}")
    
    # Create trainer config
    trainer_config = TrainerConfig(
        max_epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_grad_norm=config.max_grad_norm,
        use_ema=config.use_ema,
        ema_decay=config.ema_decay,
        use_amp=config.use_amp,
        log_every=config.log_every,
        eval_every=config.eval_every,
        checkpoint_every=config.checkpoint_every,
        checkpoint_dir=str(output_dir / "checkpoints"),
        device=config.device,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    
    # Evaluation function
    def eval_fn(model: nn.Module, step: int) -> Dict[str, float]:
        model.eval()
        with torch.no_grad():
            samples = model.sample(100)
        return {
            "samples_mean": float(samples.mean()),
            "samples_std": float(samples.std()),
        }
    
    # Visualization callback
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    
    # Get real data for visualization
    if hasattr(train_dataset, 'data'):
        # Toy datasets have .data attribute
        real_data_for_viz = train_dataset.data
    else:
        # Image datasets: extract a batch for visualization
        n_viz = min(config.n_viz_real, len(train_dataset))
        real_data_for_viz = torch.stack([train_dataset[i][0] for i in range(n_viz)])
    
    # Save clean visualization of original data (for paper)
    if real_data_for_viz is not None and len(real_data_for_viz.shape) == 2 and real_data_for_viz.shape[-1] == 2:
        save_clean_data(
            real_data_for_viz,
            save_dir=samples_dir,
            name="original_data",
            n_viz=config.n_viz_real,
        )
        logger.info(f"Saved clean original data visualization to {samples_dir}/original_data_clean.png")
    
    # Create trainer first so we can reference it in callback
    trainer = None
    
    # Get checkerboard config if applicable
    checkerboard_config = get_checkerboard_config(config.dataset)
    
    # Determine model type for visualization titles
    if getattr(config, 'use_vanilla_cnn', False):
        model_type = "Vanilla CNN"
    elif config.backbone_scale == 0.0:
        model_type = "No KG (Linear)"
    else:
        model_type = f"AdS ({config.path_type.capitalize()})"
    
    def epoch_callback(model: nn.Module, epoch: int, step: int) -> None:
        if epoch % config.viz_every_epochs == 0:
            logger.info(f"Generating visualization at epoch {epoch}...")
            ema_model = trainer.get_ema_model() if trainer is not None else None
            eval_metrics = evaluate_model(
                ema_model if ema_model is not None else model,
                epoch=epoch,
                real_data=real_data_for_viz,
                n_gen=config.n_viz_gen,
                n_real=config.n_viz_real,
                save_dir=str(samples_dir),
                checkerboard_config=checkerboard_config,
                model_type=model_type,
                compute_bpd=True,  # Enable BPD computation
                bpd_n_samples=min(100, len(real_data_for_viz)) if real_data_for_viz is not None else 50,
                bpd_n_hutchinson=1,
            )
            if "sample_plot_path" in eval_metrics:
                logger.info(f"Saved visualization to {eval_metrics['sample_plot_path']}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        config=trainer_config,
        val_dataset=val_dataset,
        eval_fn=eval_fn,
        epoch_callback=epoch_callback,
    )
    
    # Train
    logger.info("Starting training...")
    results = trainer.train()
    
    # Final evaluation with EMA model
    logger.info("Final evaluation with EMA model...")
    ema_model = trainer.get_ema_model()
    final_metrics = evaluate_model(
        ema_model,
        epoch=config.epochs,
        real_data=real_data_for_viz,
        n_gen=config.n_viz_gen,
        n_real=config.n_viz_real,
        save_dir=str(samples_dir),
        checkerboard_config=checkerboard_config,
        model_type=model_type,
        compute_bpd=True,  # Enable BPD computation for final evaluation
        bpd_n_samples=min(500, len(real_data_for_viz)) if real_data_for_viz is not None else 100,
        bpd_n_hutchinson=1,
    )
    results["final_eval"] = final_metrics
    
    # Save final metrics as dedicated JSON
    # Determine method name for JSON
    if getattr(config, 'use_vanilla_cnn', False):
        method_name = "vanilla_cnn_flow_matching"
    elif config.backbone_scale == 0.0:
        method_name = "nokg_flow_matching"
    else:
        method_name = "ads_flow_matching"
    final_metrics["method"] = method_name
    final_metrics["dataset"] = config.dataset
    final_metrics["epochs"] = config.epochs
    final_metrics["final_step"] = results["final_step"]
    with open(output_dir / "final_metrics.json", "w") as f:
        serializable = {
            k: (float(v) if hasattr(v, 'item') else v)
            for k, v in final_metrics.items()
            if not isinstance(v, (torch.Tensor,)) or hasattr(v, 'item')
        }
        json.dump(serializable, f, indent=2)
    
    # ==========================================================================
    # SAVE ALL THREE CHECKPOINTS: final_model.pt, best_model.pt, ema_model.pt
    # ==========================================================================
    checkpoints_dir = output_dir / "checkpoints"
    
    # 1. final_model.pt - training weights (NOT EMA)
    final_model_path = checkpoints_dir / "final_model.pt"
    if not final_model_path.exists():
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "final_metrics": serializable,
        }, final_model_path)
        logger.info(f"Saved final model (training weights) to {final_model_path}")
    
    # 2. best_model.pt - training weights at best metric
    best_model_path = checkpoints_dir / "best_model.pt"
    if not best_model_path.exists():
        torch.save(model.state_dict(), best_model_path)
        logger.info(f"Saved best model to {best_model_path}")
    
    # 3. ema_model.pt - EMA weights
    ema_model_path = checkpoints_dir / "ema_model.pt"
    ema_model = trainer.get_ema_model()
    if ema_model is not None and not ema_model_path.exists():
        torch.save({
            "model_state_dict": ema_model.state_dict(),
            "config": asdict(config),
            "final_metrics": serializable,
        }, ema_model_path)
        logger.info(f"Saved EMA model to {ema_model_path}")
    
    # ==========================================================================
    # CREATE STANDARDIZED RESULTS FOLDER
    # ==========================================================================
    # Generate standardized experiment name: ads_{dataset}_{path_type}_{geometry}
    # Build standardized experiment name (without timestamp for reproducible paths)
    exp_name = build_experiment_name(config, include_timestamp=False)
    
    # Determine results directory based on dataset type
    is_image = config.use_image_encoding or is_image_dataset(config.dataset)
    
    if is_image:
        results_base = Path("results_images")
    else:
        results_base = Path("results")
    
    # Create standardized folder
    std_folder = results_base / exp_name / "checkpoints"
    std_folder.mkdir(parents=True, exist_ok=True)
    
    # Copy all three models to standardized folder
    for src_name in ["final_model.pt", "best_model.pt", "ema_model.pt", "training_history.pt"]:
        src_path = checkpoints_dir / src_name
        dst_path = std_folder / src_name
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
    
    # Also copy config.json and final_metrics.json
    shutil.copy2(output_dir / "config.json", results_base / exp_name / "config.json")
    if (output_dir / "final_metrics.json").exists():
        shutil.copy2(output_dir / "final_metrics.json", results_base / exp_name / "final_metrics.json")
    
    logger.info(f"Standardized results: {results_base / exp_name}")
    logger.info(f"  Name components: {parse_experiment_name(exp_name)}")
    
    # Save final results
    def to_serializable(v):
        """Convert numpy/torch types to JSON-serializable Python types."""
        if isinstance(v, (int, float, str, bool, type(None))):
            return v
        elif isinstance(v, (np.integer,)):
            return int(v)
        elif isinstance(v, (np.floating,)):
            return float(v)
        elif isinstance(v, np.ndarray):
            return v.tolist()
        elif isinstance(v, dict):
            return {k: to_serializable(val) for k, val in v.items()}
        elif isinstance(v, (list, tuple)):
            return [to_serializable(item) for item in v]
        elif hasattr(v, 'item'):  # torch tensors
            return v.item()
        else:
            return str(v)
    
    with open(output_dir / "results.json", "w") as f:
        results_json = {k: to_serializable(v) for k, v in results.items()}
        json.dump(results_json, f, indent=2)
    
    logger.info(f"Training complete! Results saved to {output_dir}")
    
    # Print final metrics summary
    if is_image:
        logger.info(f"Final SWD: {final_metrics.get('swd', 'N/A')}")
    else:
        logger.info(f"Final Wasserstein: {final_metrics.get('wasserstein_2d', final_metrics.get('wasserstein_x', 'N/A'))}")
        logger.info(f"Final MMD: {final_metrics.get('mmd_rbf', 'N/A')}")
        # Log checkerboard metrics if available
        if 'BV' in final_metrics:
            logger.info(f"Final BV (Boundary Violation): {final_metrics['BV']:.4f}")
        if 'JS_cell' in final_metrics:
            logger.info(f"Final JS_cell: {final_metrics['JS_cell']:.6f}")
        if 'WED_norm' in final_metrics and final_metrics['WED_norm'] is not None:
            logger.info(f"Final WED_norm: {final_metrics['WED_norm']:.6f}")
        if 'CQS' in final_metrics:
            logger.info(f"Final CQS: {final_metrics['CQS']:.6f}")
    
    # Log BPD if computed
    if 'bpd' in final_metrics:
        logger.info(f"Final BPD (bits per dimension): {final_metrics['bpd']:.4f} ± {final_metrics.get('bpd_std', 0):.4f}")
        logger.info(f"Final Log-likelihood: {final_metrics.get('log_likelihood', 'N/A')}")
    
    return results


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train UV-stabilized generative flow matching model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config file
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML/JSON config file"
    )
    
    # Experiment
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    
    # Data
    all_dataset_choices = [d.value for d in DatasetType] + ["mnist", "fashion_mnist", "cifar10", "cifar100"]
    parser.add_argument(
        "--dataset", type=str, default=None,
        choices=all_dataset_choices,
        help="Dataset type"
    )
    parser.add_argument("--n_train_samples", type=int, default=None, help="Training samples")
    parser.add_argument("--n_val_samples", type=int, default=None, help="Validation samples")
    
    # Dataset-specific parameters
    parser.add_argument("--tile_n", type=int, default=None, help="Tiles for checkerboard")
    parser.add_argument("--checker_size", type=float, default=None, help="Checkerboard size")
    parser.add_argument("--checker_eps", type=float, default=None, help="Hyperboloid epsilon")
    
    # Geometry
    parser.add_argument("--d", type=int, default=None, help="Boundary dimension")
    parser.add_argument(
        "--slice_geometry", type=str, default=None,
        choices=["planar", "flat", "planar_hsv"],
        help="Slice geometry type"
    )
    parser.add_argument("--hsv_p", type=float, default=None, 
        help="HSV parameter p ∈ (0,1]: p = 1 flat, p → 0 AdS")
    parser.add_argument("--hsv_ads_threshold", type=float, default=None,
        help="Threshold for switching to AdS formulas (default: 0.05)")
    parser.add_argument("--hsv_propagator_mode", type=str, default=None,
        choices=["bessel"],
        help="HSV propagator mode: 'bessel' (closed-form conformal product)")
    
    # HSV u-bounds (stable training for ALL p)
    parser.add_argument("--hsv_use_u_bounds", action="store_true", default=None,
        help="Use u-bounds for HSV (default: True)")
    parser.add_argument("--no-hsv_use_u_bounds", action="store_false", dest="hsv_use_u_bounds",
        help="Disable u-bounds, use fixed r_ir/r_uv")
    parser.add_argument("--hsv_u_uv", type=float, default=None,
        help="HSV u at UV boundary (default: 0.1)")
    parser.add_argument("--hsv_u_ir", type=float, default=None,
        help="HSV u at IR boundary (default: 1.0)")
    
    # Bulk field
    parser.add_argument("--deltas", type=float, nargs="+", default=None, help="Conformal dimensions")
    
    # Radial domain
    parser.add_argument("--r_ir", type=float, default=None, help="IR cutoff radius")
    parser.add_argument("--r_uv", type=float, default=None, help="UV cutoff radius")
    
    # UV lift
    parser.add_argument("--lift_noise_sigma", type=float, default=None, help="Lift noise σ")
    
    # Path
    parser.add_argument(
        "--path_type", type=str, default=None,
        choices=["linear", "hermite"],
        help="Flow matching path type"
    )
    parser.add_argument("--hermite_pi_scale", type=float, default=None, help="Hermite Π̃ scale")
    parser.add_argument("--backbone_scale", type=float, default=None, help="Backbone influence")
    
    # Radial Hermite: use t-parameterised Hermite (no ω factors in tangents)
    parser.add_argument(
        "--use_radial_hermite", action="store_true", default=None,
        help="Use radial Hermite path (stable for any geometry, auto-enabled for HSV)"
    )
    parser.add_argument(
        "--no-use_radial_hermite", action="store_false", dest="use_radial_hermite",
        help="Disable radial Hermite (use affine Hermite)"
    )
    
    # Loss weighting
    parser.add_argument(
        "--use_omega_weighting", action="store_true", default=None,
        help="Weight loss by slice volume ω = f(r)^d (geometrically correct, default)"
    )
    parser.add_argument(
        "--no-use_omega_weighting", action="store_false", dest="use_omega_weighting",
        help="Disable ω weighting (unweighted MSE, numerical proxy)"
    )
    
    # Residual network
    parser.add_argument("--residual_hidden", type=int, default=None, help="Residual hidden dim (overrides cnn/mlp defaults)")
    parser.add_argument("--residual_depth", type=int, default=None, help="Residual depth (overrides cnn/mlp defaults)")
    parser.add_argument("--cnn_hidden", type=int, default=None, help="CNN hidden dim (default: 64)")
    parser.add_argument("--cnn_depth", type=int, default=None, help="CNN depth (default: 3)")
    parser.add_argument("--mlp_hidden", type=int, default=None, help="MLP hidden dim (default: 512)")
    parser.add_argument("--mlp_depth", type=int, default=None, help="MLP depth (default: 6)")
    
    # Residual type (for symplectic integration)
    parser.add_argument(
        "--residual_type", type=str, default=None,
        choices=["direct", "potential"],
        help="Residual type: 'direct' (default) or 'potential' (for symplectic)"
    )
    
    # ODE integration
    parser.add_argument(
        "--ode_solver", type=str, default=None,
        choices=["euler", "heun", "midpoint", "rk4", "leapfrog", "implicit_midpoint"],
        help="ODE solver (leapfrog/implicit_midpoint are symplectic)"
    )
    parser.add_argument("--ode_n_steps", type=int, default=None, help="ODE integration steps")
    parser.add_argument(
        "--use_affine_stepping", action="store_true",
        help="Step in affine u(r) instead of r (recommended for HSV)"
    )
    parser.add_argument(
        "--implicit_tol", type=float, default=None,
        help="Tolerance for implicit midpoint (default: 1e-6)"
    )
    parser.add_argument(
        "--implicit_max_iter", type=int, default=None,
        help="Max iterations for implicit midpoint (default: 10)"
    )
    
    # Training
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=None, help="LR warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="Gradient clipping")
    
    # EMA
    parser.add_argument("--use_ema", action="store_true", help="Use EMA")
    parser.add_argument("--no_ema", action="store_true", help="Disable EMA")
    parser.add_argument("--ema_decay", type=float, default=None, help="EMA decay")
    
    # Logging
    parser.add_argument("--log_every", type=int, default=None, help="Log frequency")
    parser.add_argument("--eval_every", type=int, default=None, help="Eval frequency")
    parser.add_argument("--checkpoint_every", type=int, default=None, help="Checkpoint frequency")
    
    # Visualization
    parser.add_argument("--n_viz_real", type=int, default=None)
    parser.add_argument("--n_viz_gen", type=int, default=None)
    parser.add_argument("--viz_every_epochs", type=int, default=None)
    
    # Hardware
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers")
    
    # ==========================================================================
    # FIELD ENCODING OPTIONS
    # ==========================================================================
    parser.add_argument("--use_field_encoding", action="store_true", help="Enable GAUSSIAN field encoding")
    parser.add_argument("--use_spectral_encoding", action="store_true", help="Enable SPECTRAL encoding")
    parser.add_argument("--spectral_n_modes", type=int, default=None, help="Fourier modes")
    parser.add_argument("--spectral_decode_resolution", type=int, default=None, help="Decode resolution")
    parser.add_argument("--spectral_decode_method", type=str, default=None, choices=["softargmax", "phase", "density"])
    parser.add_argument("--spectral_random_feature_scale", type=float, default=None)
    parser.add_argument("--use_holographic_encoding", action="store_true", help="Enable HOLOGRAPHIC grid-free")
    parser.add_argument("--use_holographic_grid", action="store_true", help="Enable HOLOGRAPHIC grid-based")
    parser.add_argument("--use_image_encoding", action="store_true", help="Enable IMAGE encoding")
    parser.add_argument("--use_vanilla_cnn", action="store_true", help="Enable VANILLA CNN flow matching (no physics)")
    
    # Shared field encoding parameters
    parser.add_argument("--field_grid_size", type=int, nargs=2, default=None)
    parser.add_argument("--field_domain", type=float, nargs=4, default=None)
    parser.add_argument("--field_sigma", type=float, default=None)
    parser.add_argument("--field_temperature", type=float, default=None)
    
    # Holographic-specific
    parser.add_argument("--holographic_regularization", type=float, default=None)
    parser.add_argument("--no_holographic_normalize", action="store_true")
    
    # ==========================================================================
    # LAPLACIAN TYPE OPTIONS
    # ==========================================================================
    parser.add_argument(
        "--laplacian_type", type=str, default=None,
        choices=["diagonal"],
        help="Laplacian implementation (diagonal FFT-based)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Keys that are CLI-only
    cli_only_keys = {
        "config", "no_ema", "use_ema", "use_amp", "no_holographic_normalize",
        "use_field_encoding", "use_spectral_encoding", "use_holographic_encoding",
        "use_holographic_grid", "use_image_encoding", "use_vanilla_cnn"
    }
    
    # Collect overrides from CLI
    overrides = {}
    for key, value in vars(args).items():
        if key in cli_only_keys:
            continue
        if key == "deltas" and value is not None:
            overrides["deltas"] = tuple(value)
        elif key == "field_grid_size" and value is not None:
            overrides["field_grid_size"] = tuple(value)
        elif key == "field_domain" and value is not None:
            overrides["field_domain"] = tuple(value)
        elif value is not None:
            overrides[key] = value
    
    # Handle boolean flags
    if args.no_ema:
        overrides["use_ema"] = False
    elif args.use_ema:
        overrides["use_ema"] = True
    
    if args.use_amp:
        overrides["use_amp"] = True
    
    if args.use_field_encoding:
        overrides["use_field_encoding"] = True
    if args.use_spectral_encoding:
        overrides["use_spectral_encoding"] = True
    if args.use_holographic_encoding:
        overrides["use_holographic_encoding"] = True
    if args.use_holographic_grid:
        overrides["use_holographic_grid"] = True
    if args.use_image_encoding:
        overrides["use_image_encoding"] = True
    if args.use_vanilla_cnn:
        overrides["use_vanilla_cnn"] = True
    if args.no_holographic_normalize:
        overrides["holographic_normalize"] = False
    
    # Load config
    config = load_config(args.config, **overrides)
    
    # Run training
    train(config)


if __name__ == "__main__":
    main()
