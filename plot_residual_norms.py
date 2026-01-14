#!/usr/bin/env python3
"""
plot_residual_norms.py

Generate residual norm vs epoch plots for comparing model convergence.

This script scans results directories for training_history.pt files and creates
plots comparing residual norms across all models for a given dataset.

Usage:
    # Single dataset:
    python plot_residual_norms.py --dataset checkerboard
    python plot_residual_norms.py --dataset mnist --results_dir results_images
    python plot_residual_norms.py --dataset checkerboard --output residual_norms.png
    
    # All datasets at once:
    python plot_residual_norms.py --all

The plot shows:
- AdS models (various geometries) with Hermite path: should have SMALL residual norms
- AdS models with Linear path: should have LARGER residual norms (learning both φ and π)
- Baseline models (spectral, MLP): should have LARGEST residual norms (no backbone)

This demonstrates that the AdS backbone is doing useful work, reducing what the
neural network needs to learn.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Plotting disabled.")


# Style settings for paper-quality figures
if MATPLOTLIB_AVAILABLE:
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 2,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


# =============================================================================
# Constants
# =============================================================================

KNOWN_GEOMETRIES = ["planar", "flat", "planar_hsv"]
KNOWN_PATH_TYPES = ["hermite", "linear"]

# Color scheme by geometry
GEOMETRY_COLORS = {
    'planar': '#1f77b4',      # Blue
    'flat': '#9467bd',        # Purple
    'planar_hsv': '#ff7f0e',  # Orange
    None: '#7f7f7f',          # Gray for baselines
}

# Linestyle by model type
MODEL_LINESTYLES = {
    'ads': '-',               # Solid for AdS
    'spectral_baseline': '--', # Dashed for spectral baseline
    'mlp_baseline': ':',      # Dotted for MLP baseline
}

# Marker by path type
PATH_MARKERS = {
    'hermite': 'o',           # Circle for Hermite
    'linear': 's',            # Square for Linear
    None: '^',                # Triangle for baselines
}


# =============================================================================
# Parsing utilities
# =============================================================================

def parse_experiment_name(exp_name: str) -> Dict[str, Any]:
    """
    Parse experiment name to extract model type, dataset, path_type, geometry.
    
    Expected formats:
    - ads_{dataset}_{path_type}_{geometry}_{timestamp}
    - ads_{dataset}_{path_type}_{timestamp} (assumes planar)
    - spectral_baseline_{dataset}_{geometry}
    - mlp_baseline_{dataset}
    
    Returns:
        Dict with keys: model_type, dataset, path_type, geometry, display_name
    """
    result = {
        'model_type': 'unknown',
        'dataset': 'unknown',
        'path_type': None,
        'geometry': None,
        'display_name': exp_name,
    }
    
    parts = exp_name.lower().split('_')
    
    if exp_name.lower().startswith('ads_'):
        result['model_type'] = 'ads'
        
        # Find geometry (if any)
        geometry = None
        geometry_idx = None
        for i, part in enumerate(parts):
            if part in KNOWN_GEOMETRIES:
                geometry = part
                geometry_idx = i
                break
        
        # Find path type (if any)
        path_type = None
        path_idx = None
        for i, part in enumerate(parts):
            if part in KNOWN_PATH_TYPES:
                path_type = part
                path_idx = i
                break
        
        result['geometry'] = geometry or 'planar'
        result['path_type'] = path_type or 'hermite'
        
        # Extract dataset (between 'ads_' and path_type or geometry)
        end_idx = min(
            path_idx if path_idx else len(parts),
            geometry_idx if geometry_idx else len(parts)
        )
        if end_idx > 1:
            result['dataset'] = '_'.join(parts[1:end_idx])
        
        result['display_name'] = f"AdS {result['geometry'].capitalize()} ({result['path_type'].capitalize()})"
    
    elif exp_name.lower().startswith('spectral_baseline_'):
        result['model_type'] = 'spectral_baseline'
        
        # Find geometry
        geometry = None
        for part in parts:
            if part in KNOWN_GEOMETRIES:
                geometry = part
                break
        
        result['geometry'] = geometry or 'planar'
        
        # Dataset is between spectral_baseline_ and geometry
        dataset_parts = []
        for i, part in enumerate(parts[2:], start=2):
            if part in KNOWN_GEOMETRIES or part.isdigit():
                break
            dataset_parts.append(part)
        result['dataset'] = '_'.join(dataset_parts) if dataset_parts else 'unknown'
        
        result['display_name'] = f"Spectral Baseline ({result['geometry'].capitalize()})"
    
    elif exp_name.lower().startswith('mlp_baseline_'):
        result['model_type'] = 'mlp_baseline'
        
        # Dataset is after mlp_baseline_
        dataset_parts = []
        for part in parts[2:]:
            if part.isdigit():
                break
            dataset_parts.append(part)
        result['dataset'] = '_'.join(dataset_parts) if dataset_parts else 'unknown'
        
        result['display_name'] = "MLP Baseline"
    
    return result


# =============================================================================
# Data loading
# =============================================================================

def load_training_history(history_path: Path) -> Optional[List[Dict]]:
    """Load training history from a checkpoint file."""
    try:
        data = torch.load(history_path, map_location='cpu', weights_only=False)
        return data.get('residual_norm_history', [])
    except Exception as e:
        print(f"Warning: Could not load {history_path}: {e}")
        return None


def find_experiments(results_dir: Path, dataset: str) -> List[Tuple[str, Path]]:
    """
    Find all experiments for a given dataset.
    
    Returns:
        List of (experiment_name, history_path) tuples
    """
    experiments = []
    
    if not results_dir.exists():
        return experiments
    
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        exp_name = exp_dir.name
        
        # Check if this experiment is for the target dataset
        if dataset.lower() not in exp_name.lower():
            continue
        
        # Look for training_history.pt
        history_path = exp_dir / "checkpoints" / "training_history.pt"
        if history_path.exists():
            experiments.append((exp_name, history_path))
        else:
            # Also check directly in experiment dir
            history_path = exp_dir / "training_history.pt"
            if history_path.exists():
                experiments.append((exp_name, history_path))
    
    return experiments


def find_all_datasets(results_dirs: List[Path]) -> Dict[str, List[Tuple[str, Path, Path]]]:
    """
    Find all datasets across multiple results directories.
    
    Returns:
        Dict mapping dataset_name -> List of (experiment_name, history_path, results_dir)
    """
    datasets: Dict[str, List[Tuple[str, Path, Path]]] = {}
    
    for results_dir in results_dirs:
        if not results_dir.exists():
            continue
            
        for exp_dir in results_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            exp_name = exp_dir.name
            info = parse_experiment_name(exp_name)
            dataset = info['dataset']
            
            if dataset == 'unknown':
                continue
            
            # Look for training_history.pt
            history_path = exp_dir / "checkpoints" / "training_history.pt"
            if not history_path.exists():
                history_path = exp_dir / "training_history.pt"
            
            if history_path.exists():
                if dataset not in datasets:
                    datasets[dataset] = []
                datasets[dataset].append((exp_name, history_path, results_dir))
    
    return datasets


# =============================================================================
# Plotting
# =============================================================================

def get_color_and_style(info: Dict) -> Tuple[str, str, str]:
    """
    Get color, linestyle, and marker for a model type.
    
    Returns:
        (color, linestyle, marker)
    """
    color = GEOMETRY_COLORS.get(info['geometry'], '#7f7f7f')
    linestyle = MODEL_LINESTYLES.get(info['model_type'], '-')
    marker = PATH_MARKERS.get(info['path_type'], '^')
    
    return color, linestyle, marker


def plot_residual_norms(
    experiments: List[Tuple[str, Path]],
    dataset: str,
    output_path: Path,
    log_scale: bool = True,
) -> None:
    """
    Create a plot comparing residual norms across all experiments.
    
    Args:
        experiments: List of (experiment_name, history_path) tuples
        dataset: Dataset name (for title)
        output_path: Where to save the plot
        log_scale: Whether to use log scale for y-axis
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot plot: matplotlib not available")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Sort experiments for consistent ordering
    def sort_key(item):
        exp_name, _ = item
        info = parse_experiment_name(exp_name)
        
        model_order = {'ads': 0, 'spectral_baseline': 1, 'mlp_baseline': 2, 'unknown': 3}
        path_order = {'hermite': 0, 'linear': 1, None: 2}
        geo_order = {'planar': 0, 'flat': 1, 'planar_hsv': 2, None: 3}
        
        return (
            model_order.get(info['model_type'], 3),
            path_order.get(info['path_type'], 2),
            geo_order.get(info['geometry'], 5),
        )
    
    experiments_sorted = sorted(experiments, key=sort_key)
    
    # Plot each experiment
    plotted_any = False
    for exp_name, history_path in experiments_sorted:
        history = load_training_history(history_path)
        if not history:
            print(f"Skipping {exp_name}: no history data")
            continue
        
        info = parse_experiment_name(exp_name)
        color, linestyle, marker = get_color_and_style(info)
        
        # Extract epochs and residual norms
        epochs = [h['epoch'] for h in history]
        norms = [h['residual_norm'] for h in history]
        
        if not epochs or not norms:
            print(f"Skipping {exp_name}: empty history")
            continue
        
        # Plot with markers every few points for visibility
        marker_every = max(1, len(epochs) // 10)
        ax.plot(
            epochs, norms,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=marker_every,
            markersize=6,
            label=info['display_name'],
        )
        plotted_any = True
    
    if not plotted_any:
        print("No experiments had valid residual norm history to plot.")
        plt.close()
        return
    
    # Formatting
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Residual Norm (MSE)')
    ax.set_title(f'Residual Norm vs Epoch — {dataset.replace("_", " ").title()} Dataset')
    
    if log_scale:
        ax.set_yscale('log')
    
    # Handle legend (avoid duplicates)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', framealpha=0.9)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved residual norm plot to: {output_path}")


def plot_residual_norms_comparison(
    experiments: List[Tuple[str, Path]],
    dataset: str,
    output_path: Path,
) -> None:
    """
    Create a multi-panel plot comparing phi and pi residual norms separately.
    
    Args:
        experiments: List of (experiment_name, history_path) tuples
        dataset: Dataset name (for title)
        output_path: Where to save the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Cannot plot: matplotlib not available")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_total, ax_phi, ax_pi = axes
    
    # Sort experiments
    def sort_key(item):
        exp_name, _ = item
        info = parse_experiment_name(exp_name)
        model_order = {'ads': 0, 'spectral_baseline': 1, 'mlp_baseline': 2}
        path_order = {'hermite': 0, 'linear': 1, None: 2}
        geo_order = {'planar': 0, 'flat': 1, 'planar_hsv': 2, None: 3}
        return (model_order.get(info['model_type'], 3), path_order.get(info['path_type'], 2), geo_order.get(info['geometry'], 5))
    
    experiments_sorted = sorted(experiments, key=sort_key)
    
    plotted_any = False
    for exp_name, history_path in experiments_sorted:
        history = load_training_history(history_path)
        if not history:
            continue
        
        info = parse_experiment_name(exp_name)
        color, linestyle, marker = get_color_and_style(info)
        
        epochs = [h['epoch'] for h in history]
        marker_every = max(1, len(epochs) // 10)
        
        # Total residual norm
        if all('residual_norm' in h for h in history):
            norms = [h['residual_norm'] for h in history]
            ax_total.plot(epochs, norms, color=color, linestyle=linestyle, marker=marker,
                         markevery=marker_every, markersize=5, label=info['display_name'])
            plotted_any = True
        
        # Phi residual norm (if available)
        if all('residual_phi_norm' in h for h in history):
            phi_norms = [h.get('residual_phi_norm', 0) for h in history]
            ax_phi.plot(epochs, phi_norms, color=color, linestyle=linestyle, marker=marker,
                       markevery=marker_every, markersize=5, label=info['display_name'])
        
        # Pi residual norm (if available)
        if all('residual_pi_norm' in h for h in history):
            pi_norms = [h.get('residual_pi_norm', 0) for h in history]
            ax_pi.plot(epochs, pi_norms, color=color, linestyle=linestyle, marker=marker,
                      markevery=marker_every, markersize=5, label=info['display_name'])
    
    if not plotted_any:
        print("No experiments had valid history to plot.")
        plt.close()
        return
    
    # Format axes
    for ax, title in zip(axes, ['Total Residual', 'Φ̃ Residual', 'Π̃ Residual']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Residual Norm (MSE)')
        ax.set_title(f'{title} — {dataset.replace("_", " ").title()}')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved detailed residual norm plot to: {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot residual norms across experiments for convergence analysis"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset name to filter experiments (e.g., checkerboard, mnist)"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Base results directory (default: results)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output filename (default: residual_norm_{dataset}.png)"
    )
    parser.add_argument(
        "--no_log_scale", action="store_true",
        help="Use linear scale instead of log scale"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate plots for all datasets found in results/ and results_images/"
    )
    parser.add_argument(
        "--detailed", action="store_true",
        help="Generate detailed 3-panel plot (total, phi, pi)"
    )
    args = parser.parse_args()
    
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for plotting")
        return
    
    # Handle --all flag
    if args.all:
        results_dirs = [Path("results"), Path("results_images"), Path("outputs")]
        all_datasets = find_all_datasets(results_dirs)
        
        if not all_datasets:
            print("No datasets found in results/, results_images/, or outputs/")
            return
        
        print(f"Found {len(all_datasets)} datasets: {list(all_datasets.keys())}")
        
        for dataset, experiments_info in all_datasets.items():
            experiments = [(name, path) for name, path, _ in experiments_info]
            _, _, results_dir = experiments_info[0]
            
            print(f"\n{'='*60}")
            print(f"Generating plot for dataset: {dataset}")
            print(f"Found {len(experiments)} experiments")
            
            output_path = results_dir / f"residual_norm_{dataset}.png"
            plot_residual_norms(
                experiments, dataset, output_path,
                log_scale=not args.no_log_scale,
            )
            
            if args.detailed:
                detailed_path = results_dir / f"residual_norm_{dataset}_detailed.png"
                plot_residual_norms_comparison(experiments, dataset, detailed_path)
        
        print(f"\n{'='*60}")
        print(f"Generated plots for {len(all_datasets)} datasets.")
        return
    
    # Single dataset mode
    if args.dataset is None:
        print("Error: Please specify --dataset or use --all")
        parser.print_help()
        return
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory does not exist: {results_dir}")
        return
    
    experiments = find_experiments(results_dir, args.dataset)
    
    if not experiments:
        print(f"No experiments found for dataset '{args.dataset}' in {results_dir}")
        return
    
    print(f"Found {len(experiments)} experiments for {args.dataset}:")
    for exp_name, path in experiments:
        info = parse_experiment_name(exp_name)
        print(f"  - {exp_name} ({info['display_name']})")
    
    output_path = Path(args.output) if args.output else results_dir / f"residual_norm_{args.dataset}.png"
    
    plot_residual_norms(
        experiments, args.dataset, output_path,
        log_scale=not args.no_log_scale,
    )
    
    if args.detailed:
        detailed_path = output_path.parent / f"residual_norm_{args.dataset}_detailed.png"
        plot_residual_norms_comparison(experiments, args.dataset, detailed_path)


if __name__ == "__main__":
    main()