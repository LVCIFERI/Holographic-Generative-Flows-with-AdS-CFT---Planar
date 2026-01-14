"""
visualization.py

Comprehensive visualization utilities for UV-stabilized generative flow matching.

This module provides:

- Sample visualization (2D point clouds, images)
- Grid and image utilities
- Radial profile plotting
- Training convergence plots (residual norms, loss curves)
- Distribution comparison plots
- Paper-quality figure settings

Document-faithful visualization for:
- Section 9: Flow matching path visualization
- Algorithm 1: Training convergence
- Algorithm 2: Sampling quality
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

Tensor = torch.Tensor

# Check for matplotlib availability
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    Figure = None
    Axes = None

# Check for PIL availability
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None


# =============================================================================
# Paper-quality figure settings
# =============================================================================

PAPER_RCPARAMS = {
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'text.usetex': False,  # Set True if LaTeX available
    'axes.linewidth': 1.2,
    'axes.grid': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
    'grid.alpha': 0.3,
}


def apply_paper_style() -> None:
    """Apply paper-quality matplotlib settings."""
    if not MATPLOTLIB_AVAILABLE:
        return
    plt.rcParams.update(PAPER_RCPARAMS)


# =============================================================================
# Color schemes
# =============================================================================

# Professional color palette
COLORS = {
    # Data visualization
    'real': '#1A5276',       # Darker professional blue
    'generated': '#78184A',  # Darker professional magenta/purple
    
    # Geometry colors
    'planar': '#1f77b4',      # Blue
    'flat': '#9467bd',        # Purple
    'planar_hsv': '#ff7f0e',  # Orange
    'default': '#7f7f7f',     # Gray
    
    # Model type colors
    'ads': '#1f77b4',
    'spectral_baseline': '#ff7f0e',
    'mlp_baseline': '#2ca02c',
}


# =============================================================================
# Grid and image utilities
# =============================================================================

def make_grid_2d(
    samples: Tensor,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = None,
) -> Tensor:
    """
    Arrange 2D samples into a grid for visualization.
    
    Args:
        samples: (B, C, H, W) or (B, H, W) tensor
        nrow: Number of images per row
        padding: Padding between images
        normalize: Whether to normalize to [0, 1]
        value_range: (min, max) for normalization
        
    Returns:
        Grid tensor (C, grid_H, grid_W)
    """
    if samples.ndim == 3:
        samples = samples.unsqueeze(1)  # Add channel dim
    
    B, C, H, W = samples.shape
    ncol = (B + nrow - 1) // nrow
    
    # Normalize
    if normalize:
        if value_range is not None:
            vmin, vmax = value_range
        else:
            vmin = samples.min()
            vmax = samples.max()
        samples = (samples - vmin) / (vmax - vmin + 1e-8)
        samples = samples.clamp(0, 1)
    
    # Create grid
    grid_H = ncol * H + (ncol + 1) * padding
    grid_W = nrow * W + (nrow + 1) * padding
    grid = torch.zeros(C, grid_H, grid_W, device=samples.device, dtype=samples.dtype)
    
    for idx in range(B):
        i = idx // nrow
        j = idx % nrow
        y = padding + i * (H + padding)
        x = padding + j * (W + padding)
        grid[:, y:y+H, x:x+W] = samples[idx]
    
    return grid


def create_image_grid(
    images: Tensor,
    nrow: int = 8,
    normalize_range: Tuple[float, float] = (-1, 1),
) -> Tuple[np.ndarray, Optional[str]]:
    """
    Create a numpy image grid for matplotlib display.
    
    Args:
        images: Image tensor (B, C, H, W) or (B, H, W)
        nrow: Number of images per row
        normalize_range: Input value range (default: [-1, 1])
        
    Returns:
        (grid_array, colormap) where colormap is 'gray' for grayscale or None for RGB
    """
    images = images.cpu()
    
    # Normalize from input range to [0, 1]
    vmin, vmax = normalize_range
    images = (images - vmin) / (vmax - vmin)
    images = images.clamp(0, 1)
    
    n = len(images)
    ncol = (n + nrow - 1) // nrow
    
    if images.ndim == 4:  # (B, C, H, W)
        C, H, W = images.shape[1:]
        if C == 1:  # Grayscale
            images = images.squeeze(1)  # (B, H, W)
            grid = np.zeros((nrow * H, ncol * W))
            for i in range(min(n, nrow * ncol)):
                r, c = i // ncol, i % ncol
                grid[r*H:(r+1)*H, c*W:(c+1)*W] = images[i].numpy()
            return grid, 'gray'
        else:  # RGB
            images = images.permute(0, 2, 3, 1)  # (B, H, W, C)
            grid = np.zeros((nrow * H, ncol * W, C))
            for i in range(min(n, nrow * ncol)):
                r, c = i // ncol, i % ncol
                grid[r*H:(r+1)*H, c*W:(c+1)*W] = images[i].numpy()
            return grid, None
    else:  # (B, H, W) - grayscale
        H, W = images.shape[1:]
        grid = np.zeros((nrow * H, ncol * W))
        for i in range(min(n, nrow * ncol)):
            r, c = i // ncol, i % ncol
            grid[r*H:(r+1)*H, c*W:(c+1)*W] = images[i].numpy()
        return grid, 'gray'


def save_samples_png(
    samples: Tensor,
    path: str,
    nrow: int = 8,
    normalize: bool = True,
) -> None:
    """
    Save samples as PNG image.
    
    Args:
        samples: Image tensor (B, C, H, W) or (B, H, W)
        path: Output file path
        nrow: Number of images per row
        normalize: Whether to normalize to [0, 1]
        
    Raises:
        ImportError: If PIL is not available
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL required for save_samples_png. Install with: pip install Pillow")
    
    grid = make_grid_2d(samples, nrow=nrow, normalize=normalize)
    
    # Convert to numpy
    if grid.shape[0] == 1:
        # Grayscale
        img_np = (grid[0].cpu().numpy() * 255).astype("uint8")
        img = Image.fromarray(img_np, mode="L")
    elif grid.shape[0] == 3:
        # RGB
        img_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        img = Image.fromarray(img_np, mode="RGB")
    else:
        raise ValueError(f"Unsupported number of channels: {grid.shape[0]}")
    
    img.save(path)


# =============================================================================
# Point cloud visualization
# =============================================================================

def plot_2d_samples(
    real_data: Optional[Tensor],
    gen_data: Tensor,
    title: str = "Samples",
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 4.5),
    xlim: Tuple[float, float] = (-4.5, 4.5),
    ylim: Tuple[float, float] = (-4.5, 4.5),
    point_size: float = 2,
    alpha: float = 0.7,
    n_viz: int = 10000,
) -> Optional[Figure]:
    """
    Plot 2D point cloud samples (real vs generated).
    
    Args:
        real_data: Real samples tensor (N, 2) or None
        gen_data: Generated samples tensor (N, 2)
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        xlim: X-axis limits
        ylim: Y-axis limits
        point_size: Point size for scatter
        alpha: Point transparency
        n_viz: Number of points to visualize
        
    Returns:
        Matplotlib Figure or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    apply_paper_style()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot real data
    if real_data is not None:
        real_subset = real_data[:n_viz].cpu().numpy()
        axes[0].scatter(
            real_subset[:, 0],
            real_subset[:, 1],
            s=point_size,
            alpha=alpha,
            c=COLORS['real'],
            edgecolors='none',
            rasterized=True,
        )
        axes[0].set_xlim(xlim)
        axes[0].set_ylim(ylim)
        axes[0].set_aspect("equal")
        axes[0].set_xlabel("$x_1$")
        axes[0].set_ylabel("$x_2$")
        axes[0].set_title(f"Ground Truth ($n={len(real_subset):,}$)")
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
    else:
        axes[0].text(0.5, 0.5, "No real data", ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title("Ground Truth")
    
    # Plot generated data
    gen_subset = gen_data[:n_viz].cpu().numpy()
    axes[1].scatter(
        gen_subset[:, 0],
        gen_subset[:, 1],
        s=point_size,
        alpha=alpha,
        c=COLORS['generated'],
        edgecolors='none',
        rasterized=True,
    )
    axes[1].set_xlim(xlim)
    axes[1].set_ylim(ylim)
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("$x_1$")
    axes[1].set_ylabel("$x_2$")
    axes[1].set_title(f"Generated ($n={len(gen_subset):,}$)")
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.suptitle(title) if title else None
    plt.tight_layout(pad=1.5)
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
        # Also save as PDF for paper
        pdf_path = Path(save_path).with_suffix('.pdf')
        fig.savefig(pdf_path, bbox_inches="tight", facecolor='white')
        plt.close(fig)
        
        # Save clean versions (no axes, no text) for paper
        save_path_obj = Path(save_path)
        
        # Save clean real data
        if real_data is not None:
            clean_real_path = save_path_obj.parent / f"{save_path_obj.stem}_clean_real.png"
            save_clean_scatter(
                real_data[:n_viz], 
                clean_real_path, 
                color=COLORS['real'],
                xlim=xlim, ylim=ylim,
                point_size=point_size, alpha=alpha
            )
        
        # Save clean generated data
        clean_gen_path = save_path_obj.parent / f"{save_path_obj.stem}_clean_generated.png"
        save_clean_scatter(
            gen_data[:n_viz], 
            clean_gen_path, 
            color=COLORS['generated'],
            xlim=xlim, ylim=ylim,
            point_size=point_size, alpha=alpha
        )
    
    return fig


def save_clean_scatter(
    data: Tensor,
    save_path: Union[str, Path],
    color: str = '#1A5276',
    xlim: Tuple[float, float] = (-4.5, 4.5),
    ylim: Tuple[float, float] = (-4.5, 4.5),
    point_size: float = 2,
    alpha: float = 0.7,
    figsize: Tuple[float, float] = (5, 5),
) -> None:
    """
    Save a clean scatter plot without axes, labels, or title.
    
    Saves both PNG and PDF versions for paper inclusion.
    
    Args:
        data: 2D point data tensor (N, 2)
        save_path: Output path (PNG, PDF will be saved with same stem)
        color: Point color
        xlim: X-axis limits
        ylim: Y-axis limits
        point_size: Point size for scatter
        alpha: Point transparency
        figsize: Figure size (should be square for equal aspect)
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    data_np = data.cpu().numpy() if hasattr(data, 'cpu') else data
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    ax.scatter(
        data_np[:, 0],
        data_np[:, 1],
        s=point_size,
        alpha=alpha,
        c=color,
        edgecolors='none',
        rasterized=True,
    )
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    
    # Remove all axes, labels, titles, etc.
    ax.axis('off')
    ax.set_frame_on(False)
    
    # Save with tight bbox and no padding
    save_path = Path(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0, 
                facecolor='white', transparent=False)
    fig.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0,
                facecolor='white', transparent=False)
    plt.close(fig)


def save_clean_data(
    data: Tensor,
    save_dir: Union[str, Path],
    name: str = "data",
    color: str = None,
    xlim: Tuple[float, float] = (-4.5, 4.5),
    ylim: Tuple[float, float] = (-4.5, 4.5),
    point_size: float = 2,
    alpha: float = 0.7,
    n_viz: int = 10000,
) -> None:
    """
    Save clean visualization of original data (no axes, no text).
    
    Call this once at the beginning of training to save clean data visualization.
    
    Args:
        data: 2D point data tensor (N, 2)
        save_dir: Directory to save to
        name: Base name for the file
        color: Point color (defaults to COLORS['real'])
        xlim: X-axis limits
        ylim: Y-axis limits
        point_size: Point size for scatter
        alpha: Point transparency
        n_viz: Number of points to visualize
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    if color is None:
        color = COLORS['real']
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = save_dir / f"{name}_clean.png"
    save_clean_scatter(
        data[:n_viz],
        save_path,
        color=color,
        xlim=xlim, ylim=ylim,
        point_size=point_size, alpha=alpha
    )


def plot_image_samples(
    real_images: Optional[Tensor],
    gen_images: Tensor,
    title: str = "Image Samples",
    save_path: Optional[str] = None,
    n_show: int = 64,
    nrow: int = 8,
) -> Optional[Figure]:
    """
    Plot image samples (real vs generated) in a grid.
    
    Args:
        real_images: Real images (B, C, H, W) or None
        gen_images: Generated images (B, C, H, W)
        title: Plot title
        save_path: Optional path to save figure
        n_show: Number of images to show
        nrow: Number of images per row in grid
        
    Returns:
        Matplotlib Figure or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    apply_paper_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Real images
    if real_images is not None:
        real_grid, cmap = create_image_grid(real_images[:n_show], nrow=nrow)
        axes[0].imshow(real_grid, cmap=cmap)
        axes[0].set_title('Real Images')
        axes[0].axis('off')
    else:
        axes[0].text(0.5, 0.5, "No real data", ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title("Real Images")
        axes[0].axis('off')
    
    # Generated images
    gen_grid, cmap = create_image_grid(gen_images[:n_show], nrow=nrow)
    axes[1].imshow(gen_grid, cmap=cmap)
    axes[1].set_title('Generated Images')
    axes[1].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        fig.savefig(Path(save_path).with_suffix('.pdf'), bbox_inches='tight')
        plt.close(fig)
    
    return fig


def save_generated_only(
    gen_images: Tensor,
    save_path: str,
    title: str = "Generated Images",
    n_show: int = 64,
    nrow: int = 8,
) -> Optional[Figure]:
    """
    Save only generated images as a clean grid (no real data comparison).
    
    Args:
        gen_images: Generated images (B, C, H, W)
        save_path: Path to save figure
        title: Plot title
        n_show: Number of images to show
        nrow: Number of images per row in grid
        
    Returns:
        Matplotlib Figure or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    apply_paper_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    gen_grid, cmap = create_image_grid(gen_images[:n_show], nrow=nrow)
    ax.imshow(gen_grid, cmap=cmap)
    ax.set_title(title)
    ax.axis('off')
    
    plt.tight_layout()
    
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    fig.savefig(Path(save_path).with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    
    return fig


# =============================================================================
# Radial profile visualization
# =============================================================================

def plot_radial_profile(
    phi_traj: Tensor,
    r_traj: Tensor,
    channel: int = 0,
    spatial_idx: Optional[Tuple[int, ...]] = None,
    ax: Optional[Any] = None,
    label: str = "Φ̃(r)",
) -> Optional[Any]:
    """
    Plot radial profile of field evolution.
    
    Document Section 3: Visualize bulk field evolution along radial direction.
    
    Args:
        phi_traj: (n_steps, B, C, *spatial)
        r_traj: (n_steps,) radii
        channel: Which channel to plot
        spatial_idx: Which spatial point to plot (default: first)
        ax: Matplotlib axis (created if None)
        label: Legend label
        
    Returns:
        Matplotlib axis
        
    Raises:
        ImportError: If matplotlib is not available
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")
    
    if ax is None:
        fig, ax = plt.subplots()
    
    r = r_traj.cpu().numpy()
    
    # Select spatial point
    if spatial_idx is None:
        # Flatten and take first point
        phi = phi_traj[:, 0, channel].reshape(len(r_traj), -1)[:, 0].cpu().numpy()
    else:
        phi = phi_traj[:, 0, channel, *spatial_idx].cpu().numpy()
    
    ax.plot(r, phi, label=label)
    ax.set_xlabel("r")
    ax.set_ylabel("Field value")
    ax.legend()
    
    return ax


# =============================================================================
# Training convergence plots
# =============================================================================

@dataclass
class ExperimentInfo:
    """Parsed experiment metadata."""
    
    model_type: str
    dataset: str
    path_type: Optional[str]
    geometry: Optional[str]
    display_name: str


def parse_experiment_name(exp_name: str) -> ExperimentInfo:
    """
    Parse experiment name to extract model type, dataset, path_type, geometry.
    
    Expected formats:
    - ads_{dataset}_{path_type}_{geometry}  (e.g., ads_checkerboard_hermite_planar)
    - spectral_baseline_{dataset}_{geometry} (e.g., spectral_baseline_checkerboard_planar)
    - mlp_baseline_{dataset} (e.g., mlp_baseline_checkerboard)
    
    Args:
        exp_name: Experiment folder name
        
    Returns:
        ExperimentInfo with parsed metadata
    """
    result = ExperimentInfo(
        model_type='unknown',
        dataset='unknown',
        path_type=None,
        geometry=None,
        display_name=exp_name,
    )
    
    parts = exp_name.split('_')
    
    if exp_name.startswith('ads_'):
        # ads_{dataset}_{path_type}_{geometry}
        result.model_type = 'ads'
        if len(parts) >= 4:
            result.dataset = parts[1]
            result.path_type = parts[2]
            result.geometry = parts[3]
            result.display_name = f"AdS {parts[3].capitalize()} ({parts[2].capitalize()})"
    
    elif exp_name.startswith('spectral_baseline_'):
        # spectral_baseline_{dataset}_{geometry}
        result.model_type = 'spectral_baseline'
        if len(parts) >= 4:
            result.dataset = parts[2]
            result.geometry = parts[3]
            result.display_name = f"Spectral Baseline ({parts[3].capitalize()})"
    
    elif exp_name.startswith('mlp_baseline_'):
        # mlp_baseline_{dataset}
        result.model_type = 'mlp_baseline'
        if len(parts) >= 3:
            result.dataset = parts[2]
            result.display_name = "MLP Baseline"
    
    return result


def get_style_for_experiment(info: ExperimentInfo) -> Tuple[str, str, str]:
    """
    Get color, linestyle, and marker for a model type.
    
    Args:
        info: Parsed experiment info
        
    Returns:
        (color, linestyle, marker)
    """
    # Color by geometry
    geometry_colors = {
        'planar': COLORS['planar'],
        'flat': COLORS['flat'],
        'planar_hsv': COLORS['planar_hsv'],
        None: COLORS['default'],
    }
    
    # Linestyle by model type
    model_linestyles = {
        'ads': '-',               # Solid for AdS
        'spectral_baseline': '--', # Dashed for spectral baseline
        'mlp_baseline': ':',      # Dotted for MLP baseline
    }
    
    # Marker by path type
    path_markers = {
        'hermite': 'o',           # Circle for Hermite
        'linear': 's',            # Square for Linear
        None: '^',                # Triangle for baselines
    }
    
    color = geometry_colors.get(info.geometry, COLORS['default'])
    linestyle = model_linestyles.get(info.model_type, '-')
    marker = path_markers.get(info.path_type, '^')
    
    return color, linestyle, marker


def load_training_history(history_path: str) -> Optional[List[Dict]]:
    """
    Load training history from a checkpoint file.
    
    Args:
        history_path: Path to training_history.pt file
        
    Returns:
        List of history dicts or None if loading fails
    """
    try:
        data = torch.load(history_path, map_location='cpu')
        return data.get('residual_norm_history', [])
    except Exception as e:
        print(f"Warning: Could not load {history_path}: {e}")
        return None


def plot_residual_norms(
    experiments: List[Tuple[str, str]],
    dataset: str,
    output_path: str,
    log_scale: bool = True,
    figsize: Tuple[float, float] = (12, 7),
) -> Optional[Figure]:
    """
    Create a plot comparing residual norms across all experiments.
    
    This demonstrates that the AdS backbone reduces what the neural network
    needs to learn (Document Algorithm 1 convergence analysis).
    
    Args:
        experiments: List of (experiment_name, history_path) tuples
        dataset: Dataset name (for title)
        output_path: Where to save the plot
        log_scale: Whether to use log scale for y-axis
        figsize: Figure size
        
    Returns:
        Matplotlib Figure or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    apply_paper_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort experiments for consistent ordering
    # Order: AdS Hermite, AdS Linear, Spectral Baseline, MLP Baseline
    def sort_key(item):
        exp_name, _ = item
        info = parse_experiment_name(exp_name)
        
        # Primary sort: model type
        model_order = {'ads': 0, 'spectral_baseline': 1, 'mlp_baseline': 2}
        
        # Secondary sort: path type (Hermite before Linear)
        path_order = {'hermite': 0, 'linear': 1, None: 2}
        
        # Tertiary sort: geometry
        geo_order = {'planar': 0, 'flat': 1, 'planar_hsv': 2, None: 3}
        
        return (
            model_order.get(info.model_type, 3),
            path_order.get(info.path_type, 2),
            geo_order.get(info.geometry, 5),
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
        color, linestyle, marker = get_style_for_experiment(info)
        
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
            label=info.display_name,
        )
        plotted_any = True
    
    if not plotted_any:
        print("No experiments had valid residual norm history to plot.")
        plt.close(fig)
        return None
    
    # Formatting
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Residual Norm (MSE)')
    ax.set_title(f'Residual Norm vs Epoch — {dataset.capitalize()} Dataset')
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved residual norm plot to: {output_path}")
    return fig


def plot_loss_curves(
    loss_history: List[float],
    val_loss_history: Optional[List[float]] = None,
    title: str = "Training Loss",
    save_path: Optional[str] = None,
    log_scale: bool = True,
) -> Optional[Figure]:
    """
    Plot training and validation loss curves.
    
    Args:
        loss_history: Training loss per step/epoch
        val_loss_history: Optional validation loss
        title: Plot title
        save_path: Optional path to save figure
        log_scale: Whether to use log scale
        
    Returns:
        Matplotlib Figure or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    apply_paper_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = list(range(len(loss_history)))
    ax.plot(steps, loss_history, label='Training Loss', color=COLORS['real'])
    
    if val_loss_history is not None:
        val_steps = list(range(len(val_loss_history)))
        ax.plot(val_steps, val_loss_history, label='Validation Loss', 
                color=COLORS['generated'], linestyle='--')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig


# =============================================================================
# Distribution comparison plots
# =============================================================================

def plot_histogram_comparison(
    real_data: Tensor,
    gen_data: Tensor,
    dim: int = 0,
    title: str = "Distribution Comparison",
    save_path: Optional[str] = None,
    bins: int = 50,
) -> Optional[Figure]:
    """
    Plot histogram comparison of real vs generated distributions.
    
    Args:
        real_data: Real samples
        gen_data: Generated samples
        dim: Which dimension to plot
        title: Plot title
        save_path: Optional path to save figure
        bins: Number of histogram bins
        
    Returns:
        Matplotlib Figure or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    apply_paper_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    real_np = real_data[:, dim].cpu().numpy()
    gen_np = gen_data[:, dim].cpu().numpy()
    
    ax.hist(real_np, bins=bins, alpha=0.6, label='Real', color=COLORS['real'], density=True)
    ax.hist(gen_np, bins=bins, alpha=0.6, label='Generated', color=COLORS['generated'], density=True)
    
    ax.set_xlabel(f'Dimension {dim}')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig


# =============================================================================
# Evaluation visualization
# =============================================================================

def create_evaluation_plot(
    real_data: Union[Tensor, np.ndarray],
    gen_data: Union[Tensor, np.ndarray],
    exp_name: str,
    checkpoint_type: str,
    save_path: str,
    is_image: bool = False,
) -> None:
    """
    Create evaluation visualization plot.
    
    Args:
        real_data: Real samples (tensor or numpy array)
        gen_data: Generated samples (tensor or numpy array)
        exp_name: Experiment name for title
        checkpoint_type: Checkpoint type (final, best, ema)
        save_path: Path to save figure
        is_image: Whether data is images
    """
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # Convert to tensor if needed
    if isinstance(real_data, np.ndarray):
        real_data = torch.from_numpy(real_data)
    if isinstance(gen_data, np.ndarray):
        gen_data = torch.from_numpy(gen_data)
    
    apply_paper_style()
    
    if is_image:
        plot_image_samples(
            real_data, gen_data,
            title=f"{exp_name} [{checkpoint_type}]",
            save_path=save_path,
        )
    else:
        plot_2d_samples(
            real_data, gen_data,
            title=f"{exp_name} [{checkpoint_type}]",
            save_path=save_path,
        )


# =============================================================================
# Metric visualization
# =============================================================================

def plot_metric_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_name: str,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Optional[Figure]:
    """
    Plot bar chart comparing a metric across experiments.
    
    Args:
        metrics_dict: Dict mapping experiment_name -> metrics dict
        metric_name: Name of metric to plot
        title: Plot title
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib Figure or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        return None
    
    apply_paper_style()
    
    # Extract data
    experiments = []
    values = []
    for exp_name, metrics in metrics_dict.items():
        if metric_name in metrics:
            experiments.append(exp_name)
            values.append(metrics[metric_name])
    
    if not experiments:
        print(f"No experiments have metric '{metric_name}'")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(experiments))
    bars = ax.bar(x, values, color=COLORS['real'])
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel(metric_name)
    ax.set_title(title or f'{metric_name} Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return fig


# =============================================================================
# Convenience functions
# =============================================================================

def visualize_samples(
    real_data: Optional[Tensor],
    gen_data: Tensor,
    epoch: int,
    save_dir: str,
    is_image: bool = False,
    n_viz: int = 10000,
) -> Dict[str, Any]:
    """
    Convenience function to visualize samples during training.
    
    Args:
        real_data: Real samples or None
        gen_data: Generated samples
        epoch: Current epoch
        save_dir: Directory to save visualizations
        is_image: Whether data is images
        n_viz: Number of samples to visualize
        
    Returns:
        Dict with visualization paths and metrics
    """
    result = {}
    
    if not MATPLOTLIB_AVAILABLE:
        return result
    
    save_path = Path(save_dir) / f"samples_epoch_{epoch:04d}.png"
    
    if is_image:
        plot_image_samples(
            real_data, gen_data,
            title=f"Epoch {epoch}",
            save_path=str(save_path),
        )
    else:
        plot_2d_samples(
            real_data, gen_data,
            title=f"Epoch {epoch}",
            save_path=str(save_path),
            n_viz=n_viz,
        )
    
    result["sample_plot_path"] = str(save_path)
    
    return result
