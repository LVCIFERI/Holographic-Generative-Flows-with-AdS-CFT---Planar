#!/usr/bin/env python3
"""
evaluate.py

Unified evaluation script for UV-stabilized generative flow matching.

Load trained models from results folder and compute metrics.
Generates visualizations and saves metrics JSON for each experiment.

Supports:
- Toy datasets (2D point clouds)
- Image datasets (MNIST, CIFAR)
- Both AdS models and baseline models (spectral_baseline, mlp_baseline)
- All THREE checkpoints: final_model.pt, best_model.pt, ema_model.pt

Now supports path_type parsing from folder names:
- ads_{dataset}_{path_type}_{geometry} (new format)
- ads_{dataset}_{geometry} (old format, assumes hermite)

Usage:
    # Evaluate toy datasets
    python evaluate.py --mode toy --results_dir results --n_samples 10000
    
    # Evaluate image datasets (MNIST only)
    python evaluate.py --mode image --results_dir results_images --n_samples 1000
    
    # Evaluate specific experiment
    python evaluate.py --experiment ads_checkerboard_hermite_planar --n_samples 10000
    
    # Filter by pattern
    python evaluate.py --mode toy --filter checkerboard --n_samples 10000
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Local imports
from ads_cft.train import (
    ExperimentConfig,
    create_model,
    parse_experiment_name,
    build_experiment_name,
    get_experiment_signature,
)
from ads_cft.data_toy import create_sampler
from ads_cft.config import DatasetType
from ads_cft.metrics import compute_metrics
from ads_cft.visualization import (
    create_evaluation_plot,
    plot_2d_samples,
    plot_image_samples,
    parse_experiment_name,
    apply_paper_style,
    MATPLOTLIB_AVAILABLE,
)

if MATPLOTLIB_AVAILABLE:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


# =============================================================================
# JSON Serialization Helper
# =============================================================================

def convert_to_serializable(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj


# =============================================================================
# Constants
# =============================================================================

# Known geometries and path types for parsing
KNOWN_GEOMETRIES = ["planar", "flat", "planar_hsv"]
KNOWN_PATH_TYPES = ["hermite", "linear"]

# Known datasets (including compound names)
KNOWN_DATASETS = [
    "checkerboard", "swiss_roll", "two_moons", "gaussian",
    "circles", "pinwheel", "mnist", "fashion_mnist", "cifar10", "cifar100",
    # Compound dataset names (must be checked before simple ones)
    "sphere_checkerboard", "hyperboloid_checkerboard",
]

# Keywords for toy dataset detection
TOY_KEYWORDS = [
    "checkerboard", "swiss_roll", "two_moons", "gaussian",
    "circles", "pinwheel", "sphere_checkerboard", "hyperboloid_checkerboard"
]

# Keywords for image dataset detection
IMAGE_KEYWORDS = ["mnist", "fashion_mnist", "cifar10", "cifar100"]

# Keywords for HSV experiment detection
HSV_KEYWORDS = ["hsv", "symplectic"]

# HSV subdirectories (run_hsv_experiments.sh creates nested structure)
HSV_SUBDIRS = [
    "planar", "planar_hsv",
    "symplectic", "residual_comparison", "u_bounds_sweep",
    "propagator_comparison", "omega_weighting", "baselines"
]


# =============================================================================
# Checkpoint utilities
# =============================================================================

def find_all_checkpoints(exp_dir: Path) -> Dict[str, Path]:
    """
    Find all available checkpoints in experiment directory.
    
    Args:
        exp_dir: Experiment directory path
        
    Returns:
        Dict mapping checkpoint_type -> checkpoint_path
    """
    checkpoints = {}
    
    # Priority locations for each checkpoint type
    checkpoint_types = {
        "final": ["checkpoints/final_model.pt", "final_model.pt"],
        "best": ["checkpoints/best_model.pt", "best_model.pt"],
        "ema": ["checkpoints/ema_model.pt", "ema_model.pt"],
    }
    
    for ckpt_type, paths in checkpoint_types.items():
        for rel_path in paths:
            full_path = exp_dir / rel_path
            if full_path.exists():
                checkpoints[ckpt_type] = full_path
                break
    
    return checkpoints


def parse_experiment_folder(exp_name: str) -> Dict[str, Any]:
    """
    Parse experiment folder name to extract configuration components.
    
    Uses the comprehensive parse_experiment_name() from train.py for new-style names,
    with fallback for legacy naming formats.
    
    New format (comprehensive):
        ads_{dataset}_{path}_{geometry}[_{hsv}][_{solver}][_{residual}][_{laplacian}][_{encoding}]
    
    Legacy formats (still supported):
        - ads_{dataset}_{path_type}_{geometry}
        - spectral_baseline_{dataset}_{geometry}
        - mlp_baseline_{dataset}
    
    Args:
        exp_name: Experiment folder name
        
    Returns:
        Dict with parsed components:
            - model_type: str (ads, mlp_baseline, spectral_baseline)
            - dataset: str
            - path_type: str or None
            - geometry: str or None
            - hsv_p: float or None
            - ode_solver: str
            - residual_type: str
            - laplacian_type: str
            - encoding: str or None
            - display_name: str (human-readable)
    """
    # Initialize result with defaults
    result = {
        'model_type': 'unknown',
        'dataset': 'unknown',
        'path_type': None,
        'geometry': None,
        'hsv_p': None,
        'ode_solver': 'rk4',
        'residual_type': 'direct',
        'laplacian_type': 'diagonal',
        'deltas': None,
        'encoding': None,
        'spectral_modes': None,
        'display_name': exp_name,
    }
    
    parts = exp_name.split('_')
    exp_lower = exp_name.lower()
    
    # Handle legacy baseline formats
    if exp_lower.startswith('mlp_baseline_'):
        # mlp_baseline_{dataset}
        result['model_type'] = 'mlp_baseline'
        result['dataset'] = "_".join(parts[2:])
        result['display_name'] = f"MLP Baseline ({result['dataset']})"
        return result
        
    elif exp_lower.startswith('spectral_baseline_'):
        # spectral_baseline_{dataset}_{geometry}
        result['model_type'] = 'spectral_baseline'
        geometry = parts[-1] if parts[-1] in KNOWN_GEOMETRIES else "planar"
        dataset_parts = parts[2:-1] if parts[-1] in KNOWN_GEOMETRIES else parts[2:]
        result['dataset'] = "_".join(dataset_parts)
        result['geometry'] = geometry
        result['display_name'] = f"Spectral Baseline ({geometry.capitalize()})"
        return result
    
    # Use comprehensive parser for ads models
    if exp_lower.startswith('ads_'):
        result['model_type'] = 'ads'
        
        # Parse manually to handle compound dataset names
        # Format: ads_{dataset}_{path}_{geometry}[_{hsv}][_{solver}][_{residual}][_{delta}][_{encoding}]
        parts = exp_name.split('_')
        
        # Remove 'ads' prefix
        remaining = parts[1:]
        
        # Check for compound dataset names first (e.g., sphere_checkerboard, hyperboloid_checkerboard)
        dataset = None
        dataset_end_idx = 0
        
        # Try compound datasets first (longer matches)
        for compound in ["sphere_checkerboard", "hyperboloid_checkerboard"]:
            compound_parts = compound.split('_')
            if len(remaining) >= len(compound_parts):
                if '_'.join(remaining[:len(compound_parts)]) == compound:
                    dataset = compound
                    dataset_end_idx = len(compound_parts)
                    break
        
        # If no compound match, use single-word dataset
        if dataset is None and remaining:
            dataset = remaining[0]
            dataset_end_idx = 1
        
        result['dataset'] = dataset or 'unknown'
        remaining = remaining[dataset_end_idx:]
        
        # Next should be path_type (hermite or linear)
        if remaining and remaining[0] in KNOWN_PATH_TYPES:
            result['path_type'] = remaining[0]
            remaining = remaining[1:]
        else:
            result['path_type'] = 'hermite'
        
        # Next should be geometry (may include hsv suffix)
        if remaining:
            geom = remaining[0]
            remaining = remaining[1:]
            
            # Check for HSV suffix (e.g., "planar" followed by "hsv")
            if remaining and remaining[0] == 'hsv':
                geom += '_hsv'
                remaining = remaining[1:]
                
                # Check for p value (e.g., "p0.5")
                if remaining and remaining[0].startswith('p'):
                    try:
                        p_val = float(remaining[0][1:])
                        result['hsv_p'] = p_val
                        remaining = remaining[1:]
                    except ValueError:
                        pass
            
            result['geometry'] = geom
        
        # Parse remaining optional parts
        for part in remaining:
            part_lower = part.lower()
            
            # ODE solver
            if part_lower in ['rk4', 'euler', 'heun', 'midpoint', 'leapfrog', 'implicit_midpoint']:
                # Handle implicit_midpoint which might be split
                result['ode_solver'] = part_lower
            elif part_lower == 'implicit' and remaining and 'midpoint' in str(remaining):
                result['ode_solver'] = 'implicit_midpoint'
            
            # Residual type
            elif part_lower in ['direct', 'potential']:
                result['residual_type'] = part_lower
            
            # Delta value (e.g., "d2.5")
            elif part_lower.startswith('d') and len(part_lower) > 1:
                try:
                    result['deltas'] = float(part_lower[1:])
                except ValueError:
                    pass
            
            # Spectral encoding (e.g., "spectral16")
            elif part_lower.startswith('spectral'):
                result['encoding'] = 'spectral'
                try:
                    result['spectral_modes'] = int(part_lower[8:])
                except ValueError:
                    pass
        
        # Build display name
        display_parts = [f"AdS {result['geometry'].capitalize() if result['geometry'] else 'Unknown'}"]
        if result['path_type']:
            display_parts.append(f"({result['path_type'].capitalize()})")
        if result['hsv_p'] is not None:
            display_parts.append(f"p={result['hsv_p']:.2f}")
        if result['ode_solver'] != 'rk4':
            display_parts.append(f"[{result['ode_solver']}]")
        if result['residual_type'] != 'direct':
            display_parts.append(f"[{result['residual_type']}]")
        if result['laplacian_type'] != 'diagonal':
            lap_str = result['laplacian_type']
            display_parts.append(f"[{lap_str}]")
        if result['deltas'] is not None:
            display_parts.append(f"[Δ={result['deltas']}]")
        if result['encoding']:
            enc_str = result['encoding']
            if result['spectral_modes']:
                enc_str += str(result['spectral_modes'])
            display_parts.append(f"[{enc_str}]")
        
        result['display_name'] = " ".join(display_parts)
        return result
    
    # Unknown format - return minimal parsing
    return result


# =============================================================================
# Model loading
# =============================================================================

def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: str = "cuda",
) -> Tuple[nn.Module, Any, Optional[str]]:
    """
    Load model from checkpoint (supports both AdS and baseline models).
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model onto
        
    Returns:
        (model, config, path_type) tuple
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    root_data_mean = None
    root_data_std = None
    
    if isinstance(checkpoint, dict) and "config" in checkpoint:
        config_dict = checkpoint["config"]
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
        root_data_mean = checkpoint.get("data_mean", None)
        root_data_std = checkpoint.get("data_std", None)
    else:
        # Try to load from config.json
        config_path = checkpoint_path.parent.parent / "config.json"
        if not config_path.exists():
            config_path = checkpoint_path.parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
        else:
            config_dict = None
        state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint
    
    # Detect model type from folder name
    exp_info = parse_experiment_folder(checkpoint_path.parent.parent.name)
    is_baseline = "baseline" in exp_info['model_type']
    is_mlp_baseline = exp_info['model_type'] == 'mlp_baseline'
    is_spectral_baseline = exp_info['model_type'] == 'spectral_baseline'
    
    # If no config, create default based on folder name
    if config_dict is None:
        print(f"  No config found, inferring from folder name")
        config_dict = {
            "dataset": exp_info['dataset'],
            "slice_geometry": exp_info['geometry'],
            "path_type": exp_info['path_type'],
            "use_spectral_encoding": True,
            "spectral_n_modes": 16,
        }
    
    # Get data shape from dataset
    if any(kw in exp_info['dataset'].lower() for kw in IMAGE_KEYWORDS):
        # Image dataset
        data_shape = (1, 28, 28) if "mnist" in exp_info['dataset'].lower() else (3, 32, 32)
    else:
        # Toy dataset - get from sampler
        sampler_kwargs = _get_sampler_kwargs(config_dict)
        sampler = create_sampler(exp_info['dataset'], **sampler_kwargs)
        data_shape = sampler.data_shape
    
    # Load model based on type
    if is_mlp_baseline or is_spectral_baseline:
        # Import baseline models
        try:
            from baselines import BaselineFlowMatching, SpectralBaselineFlowMatching
            
            if is_mlp_baseline:
                print(f"  Detected: MLP baseline")
                model = BaselineFlowMatching(
                    data_shape=data_shape,
                    hidden=config_dict.get('hidden', 512),
                    depth=config_dict.get('depth', 6),
                    time_emb_dim=config_dict.get('time_emb_dim', 128),
                    device=torch.device(device),
                )
            else:
                print(f"  Detected: Spectral baseline")
                model = SpectralBaselineFlowMatching(
                    data_dim=data_shape[0] if len(data_shape) == 1 else np.prod(data_shape),
                    n_modes=config_dict.get('spectral_n_modes', 16),
                    deltas=config_dict.get('spectral_deltas', (1.5, 2.5)),
                    hidden=config_dict.get('hidden', 64),
                    depth=config_dict.get('depth', 3),
                    time_emb_dim=config_dict.get('time_emb_dim', 128),
                    geometry=config_dict.get('spectral_geometry', 'planar'),
                    device=torch.device(device),
                )
        except ImportError:
            raise ImportError("Could not import baseline models from baselines.py")
        
        # Load state dict
        data_mean = state_dict.pop('data_mean', None) or root_data_mean
        data_std = state_dict.pop('data_std', None) or root_data_std
        model.load_state_dict(state_dict, strict=False)
        
        if data_mean is not None:
            model.data_mean = data_mean
            model.data_std = data_std
            print(f"  Data normalization loaded")
        
        path_type = None
        config = type('Config', (), config_dict)()
        
    else:
        # AdS model
        print(f"  Detected: AdS model (path_type={exp_info['path_type']})")
        config = ExperimentConfig(**{
            k: v for k, v in config_dict.items()
            if k in ExperimentConfig.__dataclass_fields__
        })
        config.device = device
        
        model = create_model(config, data_shape)
        
        # Clone tensors to avoid memory issues
        state_dict_cloned = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in state_dict.items()
        }
        
        # Extract data normalization before filtering
        data_mean = state_dict_cloned.pop('data_mean', None) or root_data_mean
        data_std = state_dict_cloned.pop('data_std', None) or root_data_std
        
        # Filter state dict
        model_keys = set(model.state_dict().keys())
        skip_keys = {
            'spectral_codec.encoder.KX', 'spectral_codec.encoder.KY',
            'spectral_codec.decoder.KX', 'spectral_codec.decoder.KY',
            'image_codec.encoder.KX', 'image_codec.encoder.KY'
        }
        state_dict_filtered = {
            k: v for k, v in state_dict_cloned.items()
            if k in model_keys and k not in skip_keys
        }
        
        model.load_state_dict(state_dict_filtered, strict=False)
        
        # Set normalization
        if data_mean is not None:
            if not isinstance(data_mean, torch.Tensor):
                data_mean = torch.tensor(data_mean, dtype=torch.float32)
            if not isinstance(data_std, torch.Tensor):
                data_std = torch.tensor(data_std, dtype=torch.float32)
            model.register_buffer("data_mean", data_mean.to(device))
            model.register_buffer("data_std", data_std.to(device))
            print(f"  Data normalization: mean={data_mean.item():.4f}, std={data_std.item():.4f}")
        
        path_type = config.path_type
    
    model.to(device)
    model.eval()
    
    return model, config, path_type


def _get_sampler_kwargs(config_dict: Dict) -> Dict:
    """Extract sampler kwargs from config dict."""
    kwargs = {}
    dataset_lower = config_dict.get('dataset', '').lower()
    
    if dataset_lower == 'checkerboard':
        if 'checker_size' in config_dict:
            kwargs['scale'] = config_dict['checker_size']
        if 'tile_n' in config_dict:
            kwargs['n_tiles'] = config_dict['tile_n']
    elif dataset_lower in ['hyperboloid_checkerboard', 'sphere_checkerboard']:
        if 'tile_n' in config_dict:
            kwargs['tile_n'] = config_dict['tile_n']
        if 'checker_size' in config_dict:
            kwargs['size'] = config_dict['checker_size']
        if dataset_lower == 'hyperboloid_checkerboard' and 'checker_eps' in config_dict:
            kwargs['eps'] = config_dict['checker_eps']
    
    return kwargs


# =============================================================================
# Sample generation
# =============================================================================

@torch.no_grad()
def generate_samples(
    model: nn.Module,
    n_samples: int,
    batch_size: int = 1000,
) -> torch.Tensor:
    """
    Generate samples in batches.
    
    Args:
        model: Generative model
        n_samples: Number of samples to generate
        batch_size: Batch size for generation
        
    Returns:
        Generated samples tensor
    """
    model.eval()
    samples_list = []
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for i in range(n_batches):
        n_batch = min(batch_size, n_samples - i * batch_size)
        samples = model.sample(n_batch)
        samples_list.append(samples.cpu())
    
    samples = torch.cat(samples_list, dim=0)
    print(f"  Sample range: [{samples.min():.3f}, {samples.max():.3f}]")
    
    return samples


# =============================================================================
# Data loading
# =============================================================================

def load_real_data(
    dataset_name: str,
    n_samples: int,
    config_dict: Optional[Dict] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Load real data for comparison.
    
    Args:
        dataset_name: Dataset name
        n_samples: Number of samples
        config_dict: Optional config dict for dataset params
        device: Device for data
        
    Returns:
        Real data tensor
    """
    dataset_lower = dataset_name.lower()
    
    if any(kw in dataset_lower for kw in IMAGE_KEYWORDS):
        # Image dataset
        if "mnist" in dataset_lower:
            from torchvision import datasets, transforms
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
            mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            indices = torch.randperm(len(mnist))[:n_samples]
            images = torch.stack([mnist[i][0] for i in indices])
            return images
        else:
            raise NotImplementedError(f"Image dataset {dataset_name} not yet supported")
    else:
        # Toy dataset
        kwargs = _get_sampler_kwargs(config_dict or {})
        sampler = create_sampler(dataset_name, **kwargs)
        return sampler.sample(n_samples, device=device)


# =============================================================================
# Single checkpoint evaluation
# =============================================================================

def evaluate_checkpoint(
    exp_dir: Path,
    ckpt_type: str,
    checkpoint_path: Path,
    n_samples: int,
    device: str,
    real_data: torch.Tensor,
    is_image: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate a single checkpoint.
    
    Args:
        exp_dir: Experiment directory
        ckpt_type: Checkpoint type (final, best, ema)
        checkpoint_path: Path to checkpoint
        n_samples: Number of samples
        device: Device to use
        real_data: Real data for comparison
        is_image: Whether this is image data
        
    Returns:
        Metrics dict or None on error
    """
    print(f"\n  --- Evaluating {ckpt_type} checkpoint ---")
    
    try:
        model, config, path_type = load_model_from_checkpoint(checkpoint_path, device)
        
        print(f"  Generating {n_samples} samples...")
        start_time = time.time()
        gen_samples = generate_samples(model, n_samples, batch_size=100 if is_image else 1000)
        gen_time = time.time() - start_time
        print(f"  Generation time: {gen_time:.2f}s")
        
        print("  Computing metrics...")
        metrics = compute_metrics(real_data[:n_samples], gen_samples, is_image=is_image)
        metrics["generation_time"] = gen_time
        metrics["n_samples"] = n_samples
        metrics["checkpoint_type"] = ckpt_type
        metrics["path_type"] = path_type if path_type else "N/A"
        
        # Print key metrics
        if is_image:
            print(f"    SWD: {metrics.get('swd', 'N/A'):.6f}")
        else:
            print(f"    Wasserstein-2D: {metrics.get('wasserstein_2d', 'N/A'):.6f}")
            print(f"    MMD (RBF): {metrics.get('mmd_rbf', 'N/A'):.6f}")
            print(f"    Coverage: {metrics.get('coverage', 'N/A'):.4f}")
        
        # Create visualization
        exp_name = exp_dir.name
        viz_path = exp_dir / f"eval_{exp_name}_{ckpt_type}.png"
        
        if is_image:
            plot_image_samples(
                real_data[:64], gen_samples[:64],
                title=f"{exp_name} [{ckpt_type}]",
                save_path=str(viz_path),
            )
        else:
            plot_2d_samples(
                real_data[:n_samples], gen_samples,
                title=f"{exp_name} [{ckpt_type}]",
                save_path=str(viz_path),
            )
        print(f"  Saved visualization: {viz_path}")
        
        # Save metrics JSON
        metrics_path = exp_dir / f"eval_metrics_{ckpt_type}.json"
        with open(metrics_path, 'w') as f:
            json.dump(convert_to_serializable(metrics), f, indent=2)
        print(f"  Saved metrics: {metrics_path}")
        
        return metrics
        
    except Exception as e:
        print(f"  ERROR evaluating {ckpt_type}: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# Experiment evaluation
# =============================================================================

def evaluate_experiment(
    exp_dir: Path,
    n_samples: int,
    device: str,
    real_data: Optional[torch.Tensor] = None,
    is_image: bool = False,
) -> List[Dict[str, Any]]:
    """
    Evaluate all checkpoints in an experiment.
    
    Args:
        exp_dir: Experiment directory
        n_samples: Number of samples
        device: Device to use
        real_data: Pre-loaded real data (None to load from config)
        is_image: Whether this is image data
        
    Returns:
        List of metrics dicts
    """
    exp_name = exp_dir.name
    print(f"\n{'='*60}")
    print(f"Evaluating: {exp_name}")
    print(f"{'='*60}")
    
    # Find checkpoints
    checkpoints = find_all_checkpoints(exp_dir)
    
    if not checkpoints:
        print(f"No checkpoints found in {exp_dir}")
        return []
    
    print(f"Found checkpoints: {list(checkpoints.keys())}")
    
    # Parse experiment info
    exp_info = parse_experiment_folder(exp_name)
    print(f"Display name: {exp_info['display_name']}")
    print(f"Dataset: {exp_info['dataset']}")
    print(f"Geometry: {exp_info['geometry']}", end="")
    if exp_info.get('hsv_p') is not None:
        print(f" (p={exp_info['hsv_p']:.2f})", end="")
    print()
    print(f"Path type: {exp_info['path_type']}")
    print(f"ODE solver: {exp_info.get('ode_solver', 'rk4')}")
    if exp_info.get('residual_type', 'direct') != 'direct':
        print(f"Residual: {exp_info['residual_type']}")
    if exp_info.get('laplacian_type', 'diagonal') != 'diagonal':
        print(f"Laplacian: {exp_info['laplacian_type']}")
    if exp_info.get('encoding'):
        enc_str = exp_info['encoding']
        if exp_info.get('spectral_modes'):
            enc_str += str(exp_info['spectral_modes'])
        print(f"Encoding: {enc_str}")
    
    # Load real data if not provided
    if real_data is None:
        # Try to get config
        config_path = exp_dir / "config.json"
        config_dict = {}
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
        
        real_data = load_real_data(exp_info['dataset'], n_samples, config_dict, device='cpu')
        print(f"Real data range: [{real_data.min():.3f}, {real_data.max():.3f}]")
    
    # Evaluate each checkpoint
    all_metrics = []
    for ckpt_type, ckpt_path in checkpoints.items():
        metrics = evaluate_checkpoint(
            exp_dir, ckpt_type, ckpt_path,
            n_samples, device, real_data, is_image
        )
        if metrics:
            # Add comprehensive experiment metadata
            metrics["experiment"] = exp_name
            metrics["dataset"] = exp_info['dataset']
            metrics["geometry"] = exp_info['geometry']
            metrics["path_type"] = exp_info.get('path_type', 'hermite')
            metrics["hsv_p"] = exp_info.get('hsv_p')
            metrics["ode_solver"] = exp_info.get('ode_solver', 'rk4')
            metrics["residual_type"] = exp_info.get('residual_type', 'direct')
            metrics["laplacian_type"] = exp_info.get('laplacian_type', 'diagonal')
            metrics["deltas"] = exp_info.get('deltas')
            metrics["encoding"] = exp_info.get('encoding')
            metrics["spectral_modes"] = exp_info.get('spectral_modes')
            metrics["display_name"] = exp_info.get('display_name', exp_name)
            all_metrics.append(metrics)
    
    # Save combined metrics
    if all_metrics:
        combined_path = exp_dir / "eval_metrics_all.json"
        with open(combined_path, 'w') as f:
            json.dump(convert_to_serializable(all_metrics), f, indent=2)
        print(f"\nSaved combined metrics: {combined_path}")
    
    return all_metrics


# =============================================================================
# Batch evaluation
# =============================================================================

def find_experiments(
    results_dirs: List[Path],
    mode: str = "all",
    filter_pattern: Optional[str] = None,
    experiment_type: str = "all",
) -> List[Path]:
    """
    Find all experiment directories.
    
    Args:
        results_dirs: List of results directories to search
        mode: "all", "toy", or "image"
        filter_pattern: Optional filter pattern
        experiment_type: "all", "standard", or "hsv"
        
    Returns:
        List of experiment directories
    """
    exp_dirs = []
    
    for results_dir in results_dirs:
        if not results_dir.exists():
            continue
        
        # Get direct subdirectories (standard experiments)
        for subdir in results_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            # Check if this is an HSV subdirectory (nested structure)
            if subdir.name in HSV_SUBDIRS:
                # Search one level deeper for HSV experiments
                for nested_dir in subdir.iterdir():
                    if nested_dir.is_dir():
                        exp_dirs.append(nested_dir)
            else:
                # Standard experiment (flat structure)
                exp_dirs.append(subdir)
    
    # Filter by experiment_type
    if experiment_type == "hsv":
        exp_dirs = [d for d in exp_dirs if any(k in d.name.lower() for k in HSV_KEYWORDS)
                    or any(k in str(d.parent.name).lower() for k in HSV_SUBDIRS)]
    elif experiment_type == "standard":
        exp_dirs = [d for d in exp_dirs if not any(k in d.name.lower() for k in HSV_KEYWORDS)
                    and d.parent.name not in HSV_SUBDIRS]
    
    # Filter by pattern
    if filter_pattern:
        exp_dirs = [d for d in exp_dirs if filter_pattern in d.name]
    
    # Filter by mode
    if mode == "toy":
        exp_dirs = [d for d in exp_dirs if any(k in d.name.lower() for k in TOY_KEYWORDS)]
    elif mode == "image":
        exp_dirs = [d for d in exp_dirs if any(k in d.name.lower() for k in IMAGE_KEYWORDS)]
        # Exclude CIFAR for now
        exp_dirs = [d for d in exp_dirs if "cifar" not in d.name.lower()]
    
    return sorted(exp_dirs)


def evaluate_all(
    results_dirs: List[Path],
    mode: str,
    n_samples: int,
    device: str,
    filter_pattern: Optional[str] = None,
    experiment_type: str = "all",
) -> List[Dict[str, Any]]:
    """
    Evaluate all experiments in results directories.
    
    Creates:
    1. Individual metrics JSON per checkpoint (eval_metrics_{type}.json)
    2. Per-experiment combined metrics (eval_metrics_all.json)
    3. Global aggregate JSON with ALL experiments (all_experiments_results.json)
    
    Args:
        results_dirs: List of results directories
        mode: "all", "toy", or "image"
        n_samples: Number of samples
        device: Device to use
        filter_pattern: Optional filter pattern
        experiment_type: "all", "standard", or "hsv"
        
    Returns:
        List of all metrics
    """
    from datetime import datetime
    
    exp_dirs = find_experiments(results_dirs, mode, filter_pattern, experiment_type)
    
    print(f"Found {len(exp_dirs)} experiment(s)")
    
    if not exp_dirs:
        print("No experiments found!")
        return []
    
    # For image mode, load real data once
    real_data_cache = {}
    is_image = mode == "image"
    
    if is_image:
        print("Loading MNIST data...")
        real_data_cache["mnist"] = load_real_data("mnist", n_samples)
        print(f"Loaded {len(real_data_cache['mnist'])} real images")
    
    all_metrics = []
    experiments_by_name = {}
    
    for exp_dir in exp_dirs:
        exp_info = parse_experiment_folder(exp_dir.name)
        
        # Get cached real data or load new
        if exp_info['dataset'] in real_data_cache:
            real_data = real_data_cache[exp_info['dataset']]
        else:
            real_data = None
        
        metrics_list = evaluate_experiment(
            exp_dir, n_samples, device, real_data, is_image
        )
        all_metrics.extend(metrics_list)
        
        # Group by experiment name
        if metrics_list:
            experiments_by_name[exp_dir.name] = {
                "path": str(exp_dir),
                "config": exp_info,
                "checkpoints": {}
            }
            for m in metrics_list:
                ckpt_type = m.get("checkpoint_type", "unknown")
                experiments_by_name[exp_dir.name]["checkpoints"][ckpt_type] = m
    
    # Create comprehensive aggregate JSON
    if all_metrics and results_dirs:
        aggregate = create_aggregate_results(
            all_metrics, 
            experiments_by_name, 
            mode, 
            n_samples, 
            results_dirs
        )
        
        # Save to first results directory
        aggregate_path = results_dirs[0] / "all_experiments_results.json"
        with open(aggregate_path, 'w') as f:
            json.dump(convert_to_serializable(aggregate), f, indent=2)
        print(f"\n{'='*60}")
        print(f"AGGREGATE RESULTS SAVED")
        print(f"{'='*60}")
        print(f"File: {aggregate_path}")
        print(f"Total experiments: {aggregate['summary']['total_experiments']}")
        print(f"Total checkpoints evaluated: {aggregate['summary']['total_checkpoints']}")
        print(f"{'='*60}")
        
        # Also save simple summary
        summary_path = results_dirs[0] / f"evaluation_summary_{mode}.json"
        with open(summary_path, 'w') as f:
            json.dump(convert_to_serializable(all_metrics), f, indent=2)
        print(f"Simple summary: {summary_path}")
    
    return all_metrics


def create_aggregate_results(
    all_metrics: List[Dict],
    experiments_by_name: Dict,
    mode: str,
    n_samples: int,
    results_dirs: List[Path],
) -> Dict[str, Any]:
    """
    Create comprehensive aggregate results JSON.
    
    Structure:
    {
        "metadata": { timestamp, mode, n_samples, results_dirs },
        "summary": { total_experiments, total_checkpoints, best_by_metric },
        "experiments": {
            "exp_name": {
                "path": "...",
                "config": { dataset, geometry, path_type, ... },
                "checkpoints": {
                    "final": { metrics... },
                    "best": { metrics... },
                    "ema": { metrics... }
                }
            }
        },
        "rankings": {
            "by_wasserstein": [...],
            "by_mmd": [...],
            "by_coverage": [...]
        }
    }
    """
    from datetime import datetime
    
    # Compute summary statistics
    total_experiments = len(experiments_by_name)
    total_checkpoints = len(all_metrics)
    
    # Find best by different metrics
    best_by_metric = {}
    
    if all_metrics:
        # Wasserstein (lower is better)
        w2_metrics = [(m.get('wasserstein_2d', float('inf')), m) for m in all_metrics if m.get('wasserstein_2d') is not None]
        if w2_metrics:
            best_w2 = min(w2_metrics, key=lambda x: x[0])
            best_by_metric['wasserstein_2d'] = {
                'value': best_w2[0],
                'experiment': best_w2[1].get('experiment'),
                'checkpoint': best_w2[1].get('checkpoint_type')
            }
        
        # MMD (lower is better)
        mmd_metrics = [(m.get('mmd_rbf', float('inf')), m) for m in all_metrics if m.get('mmd_rbf') is not None]
        if mmd_metrics:
            best_mmd = min(mmd_metrics, key=lambda x: x[0])
            best_by_metric['mmd_rbf'] = {
                'value': best_mmd[0],
                'experiment': best_mmd[1].get('experiment'),
                'checkpoint': best_mmd[1].get('checkpoint_type')
            }
        
        # Coverage (higher is better)
        cov_metrics = [(m.get('coverage', 0), m) for m in all_metrics if m.get('coverage') is not None]
        if cov_metrics:
            best_cov = max(cov_metrics, key=lambda x: x[0])
            best_by_metric['coverage'] = {
                'value': best_cov[0],
                'experiment': best_cov[1].get('experiment'),
                'checkpoint': best_cov[1].get('checkpoint_type')
            }
        
        # SWD for images (lower is better)
        swd_metrics = [(m.get('swd', float('inf')), m) for m in all_metrics if m.get('swd') is not None]
        if swd_metrics:
            best_swd = min(swd_metrics, key=lambda x: x[0])
            best_by_metric['swd'] = {
                'value': best_swd[0],
                'experiment': best_swd[1].get('experiment'),
                'checkpoint': best_swd[1].get('checkpoint_type')
            }
    
    # Create rankings
    rankings = {}
    
    # Rank by Wasserstein
    w2_ranked = sorted(
        [m for m in all_metrics if m.get('wasserstein_2d') is not None],
        key=lambda x: x.get('wasserstein_2d', float('inf'))
    )
    rankings['by_wasserstein_2d'] = [
        {
            'rank': i + 1,
            'experiment': m.get('experiment'),
            'checkpoint': m.get('checkpoint_type'),
            'geometry': m.get('geometry'),
            'path_type': m.get('path_type'),
            'value': m.get('wasserstein_2d')
        }
        for i, m in enumerate(w2_ranked[:20])  # Top 20
    ]
    
    # Rank by MMD
    mmd_ranked = sorted(
        [m for m in all_metrics if m.get('mmd_rbf') is not None],
        key=lambda x: x.get('mmd_rbf', float('inf'))
    )
    rankings['by_mmd_rbf'] = [
        {
            'rank': i + 1,
            'experiment': m.get('experiment'),
            'checkpoint': m.get('checkpoint_type'),
            'geometry': m.get('geometry'),
            'path_type': m.get('path_type'),
            'value': m.get('mmd_rbf')
        }
        for i, m in enumerate(mmd_ranked[:20])
    ]
    
    # Rank by Coverage
    cov_ranked = sorted(
        [m for m in all_metrics if m.get('coverage') is not None],
        key=lambda x: x.get('coverage', 0),
        reverse=True  # Higher is better
    )
    rankings['by_coverage'] = [
        {
            'rank': i + 1,
            'experiment': m.get('experiment'),
            'checkpoint': m.get('checkpoint_type'),
            'geometry': m.get('geometry'),
            'path_type': m.get('path_type'),
            'value': m.get('coverage')
        }
        for i, m in enumerate(cov_ranked[:20])
    ]
    
    # Build aggregate structure
    aggregate = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "evaluation_mode": mode,
            "n_samples": n_samples,
            "results_directories": [str(d) for d in results_dirs],
            "generated_by": "ads_cft.evaluate"
        },
        "summary": {
            "total_experiments": total_experiments,
            "total_checkpoints": total_checkpoints,
            "checkpoint_types": ["final", "best", "ema"],
            "best_by_metric": best_by_metric
        },
        "experiments": experiments_by_name,
        "rankings": rankings,
        "all_metrics": all_metrics  # Full list for detailed analysis
    }
    
    return aggregate
    
    return all_metrics


def print_summary(metrics: List[Dict], is_image: bool = False) -> None:
    """Print summary table of metrics."""
    if not metrics:
        return
    
    print("\n" + "="*110)
    print(f"SUMMARY (all checkpoints)")
    print("="*110)
    
    if is_image:
        print(f"{'Experiment':<50} {'Checkpoint':<10} {'Path':<8} {'SWD':>12} {'Gen Mean':>10}")
        print("-"*110)
        for m in sorted(metrics, key=lambda x: (x.get('experiment', ''), x.get('checkpoint_type', ''))):
            swd = m.get('swd', -1)
            gen_mean = m.get('gen_mean', -1)
            ckpt = m.get('checkpoint_type', 'unknown')
            path = m.get('path_type', 'N/A')
            print(f"{m['experiment']:<50} {ckpt:<10} {path:<8} {swd:>12.6f} {gen_mean:>10.4f}")
    else:
        print(f"{'Experiment':<50} {'Checkpoint':<10} {'Path':<8} {'W2':>10} {'MMD':>10} {'Cov':>8}")
        print("-"*110)
        for m in sorted(metrics, key=lambda x: (x.get('experiment', ''), x.get('checkpoint_type', ''))):
            w2 = m.get('wasserstein_2d', -1)
            mmd = m.get('mmd_rbf', -1)
            cov = m.get('coverage', -1)
            ckpt = m.get('checkpoint_type', 'unknown')
            path = m.get('path_type', 'N/A')
            print(f"{m['experiment']:<50} {ckpt:<10} {path:<8} {w2:>10.4f} {mmd:>10.6f} {cov:>8.4f}")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained models and compute metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "toy", "image"],
        help="Evaluation mode"
    )
    parser.add_argument(
        "--experiment_type", type=str, default="all",
        choices=["all", "standard", "hsv"],
        help="Experiment type: standard (flat structure), hsv (nested structure), or all"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Results directory (comma-separated or glob pattern)"
    )
    parser.add_argument(
        "--experiment", type=str, default=None,
        help="Specific experiment folder to evaluate"
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Filter experiments by pattern"
    )
    parser.add_argument(
        "--n_samples", type=int, default=10000,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Check device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    
    # Handle results directories
    results_dirs = []
    for pattern in args.results_dir.split(","):
        pattern = pattern.strip()
        matched = glob.glob(pattern)
        if matched:
            results_dirs.extend([Path(d) for d in matched])
        else:
            p = Path(pattern)
            if p.exists():
                results_dirs.append(p)
    
    # Add common directories based on mode
    if args.mode in ["all", "image"] and Path("results_images").exists():
        results_dirs.append(Path("results_images"))
    if args.mode in ["all", "toy"] and Path("results").exists():
        results_dirs.append(Path("results"))
    
    # Remove duplicates
    results_dirs = list(set(results_dirs))
    
    # Evaluate
    if args.experiment:
        # Single experiment - search both flat and nested structures
        exp_dir = None
        
        # Try flat structure first
        for rd in results_dirs:
            candidate = rd / args.experiment
            if candidate.exists():
                exp_dir = candidate
                break
        
        # Try nested HSV structure
        if exp_dir is None:
            for rd in results_dirs:
                for subdir_name in HSV_SUBDIRS:
                    candidate = rd / subdir_name / args.experiment
                    if candidate.exists():
                        exp_dir = candidate
                        break
                if exp_dir is not None:
                    break
        
        # Try as direct path
        if exp_dir is None:
            exp_dir = Path(args.experiment)
        
        if not exp_dir.exists():
            print(f"Experiment not found: {args.experiment}")
            return
        
        is_image = args.mode == "image" or any(k in args.experiment.lower() for k in IMAGE_KEYWORDS)
        metrics = evaluate_experiment(exp_dir, args.n_samples, device, is_image=is_image)
        print_summary(metrics, is_image)
    else:
        # Batch evaluation
        metrics = evaluate_all(
            results_dirs,
            args.mode,
            args.n_samples,
            device,
            args.filter,
            args.experiment_type,
        )
        print_summary(metrics, args.mode == "image")


if __name__ == "__main__":
    main()
