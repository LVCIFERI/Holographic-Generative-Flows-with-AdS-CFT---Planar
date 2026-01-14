"""
losses.py

Bulk Flow Matching Loss Functions for UV-Stabilized Generative Modelling
=========================================================================

Document-faithful implementation of:

Section 9 / Eq (fm-loss): Backbone-subtracted bulk flow matching
----------------------------------------------------------------
The training objective is:

    L_FM(θ) = E_{y~p_data, S_0~μ_IR, t~U[0,1]} [
        ||R_θ^t(S_t, t) - (U_t - V_KG^t(S_t, t))||²_{g_{r(t)}}
    ]

where:
- S_0 ~ μ_IR is the IR prior (Section 7)
- S_1 = Lift(y) is the UV lift (Section 5.1)
- S_t is the interpolated state along the path
- U_t is the target velocity (depends on path choice)
- V_KG^t is the backbone velocity in t-time: V_KG^t = dr · V_KG
- R_θ^t is the residual in t-time: R_θ^t = dr · R_θ = (0, dr · N_θ)

CRITICAL: The norm ||·||²_{g_{r(t)}} uses the INTRINSIC AdS slice measure:
    ||δS||²_{g_r} = ∫_Σ (||δΦ̃||² + ||δΠ̃||²) dvol_{g_r}

where dvol_{g_r} = ω(r) dvol_ĝ (Document eq slice-volume).

Section 9: Path options
-----------------------
1. LINEAR (robust default):
    S_t = (1-t) S_0 + t S_1
    U_t = S_1 - S_0

2. ADS_AFFINE_HERMITE (geometric):
    Uses affine parameter u(r) = (1/Z) ∫_{r_IR}^r 1/ω(s) ds
    Cubic Hermite interpolation in u-space
    Target velocity per eq (target-velocity)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ads_cft.networks import BulkState
from ads_cft.paths import linear_path_and_velocity as linear_path_interpolation

Tensor = torch.Tensor


# =============================================================================
# Note: linear_path_interpolation is imported from paths.py
# =============================================================================


# =============================================================================
# Core Loss Computation (Document eq fm-loss)
# =============================================================================


def compute_flow_matching_loss(
    S_t: BulkState,
    U_t: BulkState,
    V_KG: BulkState,
    pred_res_phi: Tensor,
    pred_res_pi: Tensor,
    r_t: Tensor,
    dr: float,
    backbone_scale: float,
    geometry,
    is_hermite_path: bool,
    use_omega_weighting: bool = True,
) -> Dict[str, Tensor]:
    """
    Compute backbone-subtracted bulk flow matching loss.

    Document eq (fm-loss):
        L_FM(θ) = E[||R_θ^t(S_t, t) - (U_t - V_KG^t(S_t, t))||²_{g_{r(t)}}]

    CRITICAL: The norm ||·||²_{g_{r(t)}} uses the intrinsic AdS slice measure:
        ||δS||²_{g_r} = ∫_Σ (||δΦ̃||² + ||δΠ̃||²) ω(r) dvol_ĝ

    This loss trains the residual network R_θ to predict the correction
    needed beyond the physics-based Klein-Gordon backbone.

    Args:
        S_t: Interpolated state at time t, shape (B, C, *spatial)
        U_t: Target velocity (dS_t/dt), same shape as S_t
        V_KG: Backbone velocity in r-time V_KG(S_t, r_t), same shape
        pred_res_phi: Predicted phi residual (dr * n_phi or zeros for Hermite)
        pred_res_pi: Predicted pi residual (dr * n_pi)
        r_t: Physical radius at time t, shape (B,) or scalar
        dr: Radial domain size (r_UV - r_IR)
        backbone_scale: Scale factor α for backbone (typically 1.0)
        geometry: AdS geometry object with slice_weight method
        is_hermite_path: Whether using Hermite path (affects total loss)
        use_omega_weighting: If True, weight MSE by slice volume ω = f(r)^d.
            If False, use unweighted MSE (numerical proxy, treats all radii equally).

    Returns:
        Dictionary containing:
        - 'total_loss': Training loss (Tensor, scalar)
        - 'loss_dphi': Phi component loss (Tensor, detached)
        - 'loss_dpi': Pi component loss (Tensor, detached)
        - 'residual_norm': Total residual norm (Tensor, detached)
        - 'residual_phi_norm': Phi residual norm (Tensor, detached)
        - 'residual_pi_norm': Pi residual norm (Tensor, detached)

    Example
    -------
    >>> losses = compute_flow_matching_loss(
    ...     S_t=state_t, U_t=target_vel, V_KG=backbone_vel,
    ...     pred_res_phi=zeros, pred_res_pi=residual,
    ...     r_t=radius, dr=1.0, backbone_scale=1.0,
    ...     geometry=geometry, is_hermite_path=True
    ... )
    >>> loss = losses['total_loss']
    >>> loss.backward()

    Notes
    -----
    Path-dependent loss structure:

    HERMITE path (is_hermite_path=True):
        - R_θ = (0, N_θ) - residual only affects dΠ̃/dr
        - Only train pi component (phi residual is always 0)
        - The Hermite construction ensures Π̃_t = ∂_r Φ̃_t, which matches
          the backbone's dΦ̃/dr = Π̃

    LINEAR path (is_hermite_path=False):
        - R_θ = (N_θ^φ, N_θ^π) - full residual
        - Train both phi and pi components
    """
    device = S_t.phi_tilde.device
    dtype = S_t.phi_tilde.dtype

    # Convert backbone to t-time: V_KG^t = α · dr · V_KG
    # This scales from r-derivative to t-derivative
    V_KG_t = BulkState(
        phi_tilde=backbone_scale * dr * V_KG.phi_tilde,
        pi_tilde=backbone_scale * dr * V_KG.pi_tilde,
    )

    # Target residual: what the residual network should predict
    # target_res = U_t - V_KG^t
    target_res_phi = U_t.phi_tilde - V_KG_t.phi_tilde
    target_res_pi = U_t.pi_tilde - V_KG_t.pi_tilde

    # Compute squared errors: (predicted - target)²
    sq_err_phi = (pred_res_phi - target_res_phi) ** 2
    sq_err_pi = (pred_res_pi - target_res_pi) ** 2

    # Slice measure weighting (Document eq fm-loss)
    # ||·||²_{g_{r(t)}} = ∫ (·)² dvol_{g_r} = ∫ (·)² ω(r) dvol_ĝ
    # where ω(r) = f(r)^d is the slice weight
    if use_omega_weighting:
        omega = geometry.slice_weight(r_t)  # (B,) = f(r)^d

        # Ensure omega is on correct device and dtype
        if hasattr(omega, "to"):
            omega = omega.to(device=device, dtype=dtype)
        elif not isinstance(omega, Tensor):
            omega = torch.tensor(omega, device=device, dtype=dtype)

        # Broadcast omega to match state shape
        # From (B,) to (B, 1, 1, ...) for spatial broadcasting
        omega_b = omega
        while omega_b.ndim < S_t.phi_tilde.ndim:
            omega_b = omega_b.unsqueeze(-1)

        # Apply omega weighting to squared errors
        weighted_sq_err_phi = omega_b * sq_err_phi
        weighted_sq_err_pi = omega_b * sq_err_pi
    else:
        # Unweighted MSE (proxy loss, treats all radii equally)
        weighted_sq_err_phi = sq_err_phi
        weighted_sq_err_pi = sq_err_pi

    # Integrate over spatial dimensions (if any)
    spatial_dims = tuple(range(2, S_t.phi_tilde.ndim))
    if len(spatial_dims) > 0:
        # Field data: sum over spatial, mean over channels
        loss_phi_per_sample = weighted_sq_err_phi.sum(dim=spatial_dims).mean(dim=1)
        loss_pi_per_sample = weighted_sq_err_pi.sum(dim=spatial_dims).mean(dim=1)
    else:
        # Point data: mean over channels
        loss_phi_per_sample = weighted_sq_err_phi.mean(dim=1)
        loss_pi_per_sample = weighted_sq_err_pi.mean(dim=1)

    # Average over batch
    loss_dphi = loss_phi_per_sample.mean()
    loss_dpi = loss_pi_per_sample.mean()

    # Total loss depends on path type (DOCUMENT-FAITHFUL)
    if is_hermite_path:
        # HERMITE: Only train pi component (phi residual is always 0)
        # Document Section 4.2: R_θ = (0, N_θ)
        total_loss = loss_dpi
    else:
        # LINEAR: Train both components
        total_loss = loss_dpi + loss_dphi

    # Compute residual norms for tracking (independent of loss)
    # This measures how much the network has to correct beyond the backbone
    with torch.no_grad():
        res_phi_norm = (pred_res_phi ** 2).mean()
        res_pi_norm = (pred_res_pi ** 2).mean()
        # Combined residual norm
        if is_hermite_path:
            residual_norm = res_pi_norm
        else:
            residual_norm = res_phi_norm + res_pi_norm

    return {
        "total_loss": total_loss,
        "loss_dphi": loss_dphi.detach(),
        "loss_dpi": loss_dpi.detach(),
        "residual_norm": residual_norm.detach(),
        "residual_phi_norm": res_phi_norm.detach(),
        "residual_pi_norm": res_pi_norm.detach(),
    }


# =============================================================================
# Additional Loss Functions
# =============================================================================


def compute_mse_loss(
    pred: Tensor,
    target: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """
    Compute mean squared error loss.

    Args:
        pred: Predicted tensor
        target: Target tensor
        reduction: Reduction method ('mean', 'sum', 'none')

    Returns:
        MSE loss
    """
    return F.mse_loss(pred, target, reduction=reduction)


def compute_weighted_mse_loss(
    pred: Tensor,
    target: Tensor,
    weights: Tensor,
    reduction: str = "mean",
) -> Tensor:
    """
    Compute weighted mean squared error loss.

    Args:
        pred: Predicted tensor
        target: Target tensor
        weights: Weight tensor (broadcastable to pred shape)
        reduction: Reduction method ('mean', 'sum', 'none')

    Returns:
        Weighted MSE loss
    """
    sq_err = (pred - target) ** 2
    weighted_sq_err = weights * sq_err

    if reduction == "mean":
        return weighted_sq_err.mean()
    elif reduction == "sum":
        return weighted_sq_err.sum()
    else:
        return weighted_sq_err


def compute_sobolev_loss(
    pred: Tensor,
    target: Tensor,
    laplacian_op,
    alpha: float = 1.0,
) -> Tensor:
    """
    Compute Sobolev-weighted loss incorporating Laplacian smoothness.

    The Sobolev loss encourages smoothness by penalizing high-frequency
    errors more heavily:

        L = ||pred - target||² + α||-Δ(pred - target)||²

    Args:
        pred: Predicted field, shape (B, C, *spatial)
        target: Target field, same shape
        laplacian_op: Laplacian operator with apply_minus_laplacian method
        alpha: Weight for Laplacian term (default: 1.0)

    Returns:
        Sobolev loss (scalar)
    """
    diff = pred - target

    # L² term
    l2_loss = (diff ** 2).mean()

    # H¹ term (Laplacian)
    if laplacian_op is not None and alpha > 0:
        lap_diff = laplacian_op.apply_minus_laplacian(diff)
        h1_loss = (lap_diff ** 2).mean()
        return l2_loss + alpha * h1_loss

    return l2_loss


def compute_spectral_loss(
    pred: Tensor,
    target: Tensor,
    eigenvalues: Tensor,
    s: float = 1.0,
) -> Tensor:
    """
    Compute spectrally-weighted loss in eigenvalue basis.

    Applies Sobolev-type weighting in the spectral domain:

        L = Σ_j (1 + λ_j)^s |pred_j - target_j|²

    Args:
        pred: Predicted coefficients in spectral basis
        target: Target coefficients
        eigenvalues: Laplacian eigenvalues λ_j
        s: Sobolev smoothing exponent (default: 1.0)

    Returns:
        Spectrally-weighted loss (scalar)
    """
    diff = pred - target
    weights = (1.0 + eigenvalues) ** s

    # Broadcast weights
    while weights.ndim < diff.ndim:
        weights = weights.unsqueeze(0)

    weighted_diff = weights * (diff ** 2)
    return weighted_diff.mean()


# =============================================================================
# Loss Tracking Utilities
# =============================================================================


class LossTracker:
    """
    Track loss statistics during training.

    Provides running averages and history for loss monitoring.

    Attributes
    ----------
    window_size : int
        Window size for running average
    history : Dict[str, List[float]]
        Full history of all losses

    Example
    -------
    >>> tracker = LossTracker(window_size=100)
    >>> for step in range(1000):
    ...     losses = model.compute_losses(batch)
    ...     tracker.update(losses, step)
    ...     if step % 100 == 0:
    ...         print(tracker.get_averages())
    """

    def __init__(self, window_size: int = 100) -> None:
        """
        Initialize loss tracker.

        Args:
            window_size: Window size for running average
        """
        self.window_size = window_size
        self.history: Dict[str, list] = {}
        self._windows: Dict[str, list] = {}

    def update(self, losses: Dict[str, Tensor], step: int) -> None:
        """
        Update tracker with new losses.

        Args:
            losses: Dictionary of loss tensors
            step: Current training step
        """
        for name, value in losses.items():
            if isinstance(value, Tensor):
                value = value.item()

            if name not in self.history:
                self.history[name] = []
                self._windows[name] = []

            self.history[name].append((step, value))
            self._windows[name].append(value)

            # Keep window size
            if len(self._windows[name]) > self.window_size:
                self._windows[name].pop(0)

    def get_averages(self) -> Dict[str, float]:
        """
        Get running averages of all tracked losses.

        Returns:
            Dictionary of average values
        """
        return {
            name: sum(window) / len(window) if window else 0.0
            for name, window in self._windows.items()
        }

    def get_latest(self) -> Dict[str, float]:
        """
        Get latest values of all tracked losses.

        Returns:
            Dictionary of latest values
        """
        return {
            name: window[-1] if window else 0.0
            for name, window in self._windows.items()
        }

    def reset(self) -> None:
        """Reset all tracking data."""
        self.history.clear()
        self._windows.clear()
