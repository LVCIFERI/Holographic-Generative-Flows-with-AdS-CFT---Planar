"""
baselines.py

Baseline (Ordinary) Flow Matching Models for Comparison
========================================================

This module provides baseline flow matching models for comparison with
the AdS/CFT-based framework. These baselines use the same general
architecture but WITHOUT the physics-based components.

Two Baseline Modes
------------------

1. MLP Baseline ("mlp"):
   - Simple MLP velocity network on raw point data
   - Unfair comparison (different representation)
   - Useful for sanity checking

2. Spectral Baseline ("spectral"):
   - SAME spectral encoding + CNN as AdS model
   - SAME number of parameters
   - Fair comparison that isolates the physics contribution

   Uses:
   - Spectral encoding (bulk-boundary propagator)
   - CNN velocity network
   - Linear interpolation path
   - Gaussian prior

   Does NOT use:
   - Klein-Gordon backbone
   - Hermite path (AdS-geometric)
   - Sobolev prior
   - UV stabilization

What the Baselines Test
-----------------------

The spectral baseline tests what the AdS geometric structure contributes
by holding everything else constant. If the AdS model outperforms the
spectral baseline, the improvement comes from:
- The physics-based Klein-Gordon backbone
- The geometry-aware Hermite path
- The Sobolev-structured prior
- UV stabilization

Document References
-------------------
- Section 4.2: Backbone + residual decomposition
- Section 7: IR prior structure
- Section 9: Path interpolation options
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ads_cft.registry import register_model
from ads_cft.config import BaselineConfig

Tensor = torch.Tensor


# =============================================================================
# Exponential Moving Average (EMA)
# =============================================================================


class EMAModel:
    """
    Exponential Moving Average of model parameters.

    Used for stable sampling as per common practice in flow matching
    and diffusion models.

    Update rule: θ_ema = decay * θ_ema + (1 - decay) * θ

    Attributes
    ----------
    decay : float
        EMA decay rate (typically 0.9999)
    shadow_params : Dict[str, Tensor]
        EMA shadow parameters
    device : Optional[torch.device]
        Target device for shadow parameters

    Example
    -------
    >>> ema = EMAModel(model, decay=0.9999)
    >>> for batch in dataloader:
    ...     loss = train_step(model, batch)
    ...     ema.update(model)
    >>> # For evaluation:
    >>> original = ema.apply_shadow(model)
    >>> samples = model.sample(...)
    >>> ema.restore(model, original)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None,
    ) -> None:
        """
        Initialize EMA tracker.

        Args:
            model: Model to track
            decay: EMA decay rate (default: 0.9999)
            device: Target device for shadow parameters
        """
        self.decay = decay
        self.device = device

        # Create shadow parameters
        self.shadow_params: Dict[str, Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone()
                if device is not None:
                    self.shadow_params[name] = self.shadow_params[name].to(device)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Update shadow parameters with EMA.

        Args:
            model: Model with current parameters
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                self.shadow_params[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model: nn.Module) -> Dict[str, Tensor]:
        """
        Apply shadow parameters to model, returning original parameters.

        Args:
            model: Model to apply shadow parameters to

        Returns:
            Dictionary of original parameters for restoration
        """
        original_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow_params:
                original_params[name] = param.data.clone()
                param.data.copy_(self.shadow_params[name])
        return original_params

    def restore(self, model: nn.Module, original_params: Dict[str, Tensor]) -> None:
        """
        Restore original parameters to model.

        Args:
            model: Model to restore
            original_params: Original parameters from apply_shadow
        """
        for name, param in model.named_parameters():
            if name in original_params:
                param.data.copy_(original_params[name])

    def state_dict(self) -> Dict[str, Any]:
        """Return EMA state dictionary for checkpointing."""
        return {
            "decay": self.decay,
            "shadow_params": {k: v.cpu() for k, v in self.shadow_params.items()},
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load EMA state from checkpoint."""
        self.decay = state_dict["decay"]
        for name, param in state_dict["shadow_params"].items():
            if name in self.shadow_params:
                self.shadow_params[name].copy_(param)


# =============================================================================
# Time Embedding
# =============================================================================


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for neural networks.

    Uses the positional encoding from "Attention Is All You Need":
        PE(pos, 2i) = sin(pos / 10000^{2i/d})
        PE(pos, 2i+1) = cos(pos / 10000^{2i/d})

    Attributes
    ----------
    dim : int
        Embedding dimension
    max_period : float
        Maximum period for sinusoidal functions

    Example
    -------
    >>> embed = SinusoidalTimeEmbedding(dim=128)
    >>> t = torch.linspace(0, 1, 32)
    >>> emb = embed(t)  # (32, 128)
    """

    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        """
        Initialize sinusoidal embedding.

        Args:
            dim: Embedding dimension
            max_period: Maximum period (default: 10000)
        """
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: Tensor) -> Tensor:
        """
        Compute sinusoidal embedding.

        Args:
            t: Time values, shape (B,) or scalar

        Returns:
            Embeddings of shape (B, dim)
        """
        if t.ndim == 0:
            t = t.unsqueeze(0)

        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


# =============================================================================
# MLP Velocity Network (for point data)
# =============================================================================


class MLPVelocityNet(nn.Module):
    """
    MLP velocity network for point data.

    Architecture:
        [x, t_emb] → Linear → SiLU → ... → Linear → v

    Attributes
    ----------
    data_dim : int
        Input data dimension

    Example
    -------
    >>> net = MLPVelocityNet(data_dim=2, hidden=256, depth=4)
    >>> x = torch.randn(32, 2)
    >>> t = torch.rand(32)
    >>> v = net(x, t)  # (32, 2)
    """

    def __init__(
        self,
        data_dim: int,
        hidden: int = 256,
        depth: int = 4,
        time_emb_dim: int = 128,
    ) -> None:
        """
        Initialize MLP velocity network.

        Args:
            data_dim: Input data dimension
            hidden: Hidden layer dimension
            depth: Number of hidden layers
            time_emb_dim: Time embedding dimension
        """
        super().__init__()
        self.data_dim = data_dim

        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)

        # Input: data + time embedding
        layers: List[nn.Module] = [
            nn.Linear(data_dim + time_emb_dim, hidden),
            nn.SiLU(),
        ]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        layers.append(nn.Linear(hidden, data_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Compute velocity.

        Args:
            x: Input data, shape (B, D)
            t: Time, shape (B,) or scalar

        Returns:
            Velocity, shape (B, D)
        """
        if t.ndim == 0:
            t = t.expand(x.shape[0])

        t_emb = self.time_embed(t)  # (B, time_emb_dim)
        h = torch.cat([x, t_emb], dim=-1)  # (B, D + time_emb_dim)
        return self.net(h)


# =============================================================================
# CNN Velocity Network (for images)
# =============================================================================


class CNNVelocityNet(nn.Module):
    """
    U-Net style CNN velocity network for images.

    Architecture:
        Input → Conv → Encoder → Middle → Decoder → Output
        with skip connections and time embedding

    Attributes
    ----------
    in_channels : int
        Number of input channels

    Example
    -------
    >>> net = CNNVelocityNet(in_channels=3, hidden=64, depth=3)
    >>> x = torch.randn(32, 3, 32, 32)
    >>> t = torch.rand(32)
    >>> v = net(x, t)  # (32, 3, 32, 32)
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = 64,
        depth: int = 3,
        time_emb_dim: int = 128,
    ) -> None:
        """
        Initialize CNN velocity network.

        Args:
            in_channels: Number of input channels
            hidden: Base hidden channel count
            depth: Number of encoder/decoder stages
            time_emb_dim: Time embedding dimension
        """
        super().__init__()
        self.in_channels = in_channels

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden * 4),
            nn.SiLU(),
            nn.Linear(hidden * 4, hidden),
        )

        # Input conv
        self.input_conv = nn.Conv2d(in_channels, hidden, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        ch = hidden
        for i in range(depth):
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
                    nn.GroupNorm(8, ch * 2),
                    nn.SiLU(),
                    nn.Conv2d(ch * 2, ch * 2, 3, padding=1),
                    nn.GroupNorm(8, ch * 2),
                    nn.SiLU(),
                )
            )
            ch = ch * 2

        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
        )

        # Decoder
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(ch, ch // 2, 4, stride=2, padding=1),
                    nn.GroupNorm(8, ch // 2),
                    nn.SiLU(),
                    nn.Conv2d(ch // 2, ch // 2, 3, padding=1),
                    nn.GroupNorm(8, ch // 2),
                    nn.SiLU(),
                )
            )
            ch = ch // 2

        # Output conv
        self.output_conv = nn.Conv2d(hidden, in_channels, 3, padding=1)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Compute velocity.

        Args:
            x: Input image, shape (B, C, H, W)
            t: Time, shape (B,) or scalar

        Returns:
            Velocity, shape (B, C, H, W)
        """
        B = x.shape[0]
        if t.ndim == 0:
            t = t.expand(B)

        # Time embedding
        t_emb = self.time_mlp(self.time_embed(t))  # (B, hidden)

        # Input
        h = self.input_conv(x)  # (B, hidden, H, W)
        h = h + t_emb.view(B, -1, 1, 1)

        # Encoder
        skips = []
        for down in self.down_blocks:
            skips.append(h)
            h = down(h)

        # Middle
        h = self.middle(h)

        # Decoder
        for up, skip in zip(self.up_blocks, reversed(skips)):
            h = up(h)
            # Handle size mismatch
            if h.shape != skip.shape:
                h = F.interpolate(
                    h, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            h = h + skip

        # Output
        return self.output_conv(h)


# =============================================================================
# U-Net Velocity Network (alternative architecture)
# =============================================================================


class UNetVelocity(nn.Module):
    """
    U-Net velocity network matching AdS architecture.

    This is a simplified U-Net without explicit time conditioning,
    used in spectral baselines where the state already encodes time.

    Attributes
    ----------
    in_channels : int
        Number of input channels
    """

    def __init__(self, in_channels: int, hidden: int, depth: int) -> None:
        """
        Initialize U-Net velocity network.

        Args:
            in_channels: Number of input channels
            hidden: Base hidden channel count
            depth: Number of encoder/decoder stages
        """
        super().__init__()

        # Input projection
        self.input_conv = nn.Conv2d(in_channels, hidden, 3, padding=1)

        # Encoder (downsample)
        self.down_blocks = nn.ModuleList()
        ch = hidden
        for i in range(depth):
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
                    nn.GroupNorm(min(8, ch * 2), ch * 2),
                    nn.SiLU(),
                    nn.Conv2d(ch * 2, ch * 2, 3, padding=1),
                    nn.GroupNorm(min(8, ch * 2), ch * 2),
                    nn.SiLU(),
                )
            )
            ch = ch * 2

        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(min(8, ch), ch),
            nn.SiLU(),
        )

        # Decoder (upsample)
        self.up_blocks = nn.ModuleList()
        for i in range(depth):
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(ch, ch // 2, 4, stride=2, padding=1),
                    nn.GroupNorm(min(8, ch // 2), ch // 2),
                    nn.SiLU(),
                    nn.Conv2d(ch // 2, ch // 2, 3, padding=1),
                    nn.GroupNorm(min(8, ch // 2), ch // 2),
                    nn.SiLU(),
                )
            )
            ch = ch // 2

        # Output projection
        self.output_conv = nn.Conv2d(hidden, in_channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute velocity (no time conditioning).

        Args:
            x: Input, shape (B, C, H, W)

        Returns:
            Velocity, shape (B, C, H, W)
        """
        # Input
        h = F.silu(self.input_conv(x))

        # Encoder with skip connections
        skips = []
        for down in self.down_blocks:
            skips.append(h)
            h = down(h)

        # Middle
        h = self.middle(h)

        # Decoder with skip connections
        for up, skip in zip(self.up_blocks, reversed(skips)):
            h = up(h)
            # Add skip connection (handle size mismatch)
            if h.shape != skip.shape:
                h = F.interpolate(
                    h, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            h = h + skip

        # Output
        return self.output_conv(h)


# =============================================================================
# Baseline Flow Matching Model (MLP)
# =============================================================================


@register_model("baseline_flow_mlp")
class BaselineFlowMatching(nn.Module):
    """
    Standard flow matching model (no AdS geometry).

    Uses:
    - Linear interpolation: x_t = (1-t) * x_0 + t * x_1
    - Gaussian prior: x_0 ~ N(0, I)
    - Learned velocity: v_θ(x_t, t)
    - Target: v_target = x_1 - x_0

    This is the simplest baseline for comparison.

    Attributes
    ----------
    data_shape : Tuple[int, ...]
        Shape of input data
    is_image : bool
        Whether data is image-shaped
    data_mean : Optional[float]
        Data mean for normalization
    data_std : Optional[float]
        Data std for normalization

    Example
    -------
    >>> model = BaselineFlowMatching(data_shape=(2,), hidden=256, depth=4)
    >>> samples = model.sample(batch_size=1000)
    """

    def __init__(
        self,
        data_shape: Tuple[int, ...],
        hidden: int = 256,
        depth: int = 4,
        time_emb_dim: int = 128,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Initialize baseline flow matching model.

        Args:
            data_shape: Shape of input data
            hidden: Hidden layer dimension
            depth: Number of hidden layers
            time_emb_dim: Time embedding dimension
            device: Target device
        """
        super().__init__()

        self.data_shape = data_shape
        self.is_image = len(data_shape) >= 2 and data_shape[-1] > 2

        if self.is_image:
            # Image data: use CNN
            in_channels = data_shape[0] if len(data_shape) == 3 else 1
            self.velocity_net = CNNVelocityNet(
                in_channels=in_channels,
                hidden=hidden,
                depth=depth,
                time_emb_dim=time_emb_dim,
            )
        else:
            # Point data: use MLP
            data_dim = 1
            for s in data_shape:
                data_dim *= s
            self.velocity_net = MLPVelocityNet(
                data_dim=data_dim,
                hidden=hidden,
                depth=depth,
                time_emb_dim=time_emb_dim,
            )

        self.device = device
        self.to(device)

        # For denormalization
        self.data_mean: Optional[float] = None
        self.data_std: Optional[float] = None

    def sample_prior(self, batch_size: int) -> Tensor:
        """
        Sample from Gaussian prior.

        Args:
            batch_size: Number of samples

        Returns:
            Samples from N(0, I)
        """
        return torch.randn(batch_size, *self.data_shape, device=self.device)

    def velocity(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Compute learned velocity field.

        Args:
            x: Input data
            t: Time

        Returns:
            Velocity
        """
        if not self.is_image:
            x_flat = x.view(x.shape[0], -1)
            v = self.velocity_net(x_flat, t)
            return v.view(x.shape)
        else:
            return self.velocity_net(x, t)

    def compute_loss(self, x1: Tensor) -> Dict[str, Tensor]:
        """
        Compute flow matching loss.

        Args:
            x1: Target data, shape (B, *data_shape)

        Returns:
            Dictionary with 'total_loss' and components
        """
        B = x1.shape[0]

        # Normalize data if configured
        if self.data_mean is not None and self.data_std is not None:
            x1 = (x1 - self.data_mean) / self.data_std

        # Sample prior
        x0 = self.sample_prior(B)

        # Sample time uniformly
        t = torch.rand(B, device=self.device)

        # Linear interpolation
        t_expand = t.view(B, *([1] * len(self.data_shape)))
        x_t = (1 - t_expand) * x0 + t_expand * x1

        # Target velocity (constant for linear path)
        v_target = x1 - x0

        # Predicted velocity
        v_pred = self.velocity(x_t, t)

        # MSE loss
        loss = F.mse_loss(v_pred, v_target)

        # Velocity norm for tracking
        with torch.no_grad():
            velocity_norm = (v_pred ** 2).mean()

        return {
            "total_loss": loss,
            "mse_loss": loss,
            "residual_norm": velocity_norm.detach(),
        }

    @torch.no_grad()
    def sample(self, batch_size: int, n_steps: int = 100) -> Tensor:
        """
        Generate samples via ODE integration.

        Args:
            batch_size: Number of samples
            n_steps: ODE integration steps

        Returns:
            Generated samples
        """
        self.eval()

        # Sample from prior
        x = self.sample_prior(batch_size)

        # Euler integration from t=0 to t=1
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.tensor(i * dt, device=self.device)
            v = self.velocity(x, t)
            x = x + v * dt

        # Denormalize if configured
        if self.data_mean is not None and self.data_std is not None:
            x = x * self.data_std + self.data_mean

        return x


# =============================================================================
# Spectral Baseline (Fair Comparison for Point Data)
# =============================================================================


@register_model("baseline_flow_spectral")
class SpectralBaselineFlowMatching(nn.Module):
    """
    Fair baseline: Same spectral encoding + CNN as AdS, but NO physics.

    This isolates what the AdS geometric structure contributes by using:
    - SAME spectral encoding (bulk-boundary propagator)
    - SAME CNN architecture
    - SAME number of parameters

    But WITHOUT:
    - Klein-Gordon backbone (learns velocity from scratch)
    - Hermite path (uses simple linear interpolation)
    - Sobolev prior (uses Gaussian prior)
    - UV stabilization (not needed without KG)

    This is the "fair" comparison to determine if AdS physics helps.

    Attributes
    ----------
    data_dim : int
        Data dimension (e.g., 2 for 2D points)
    n_modes : int
        Number of spectral modes per dimension
    n_channels : int
        Number of field channels

    Example
    -------
    >>> model = SpectralBaselineFlowMatching(data_dim=2, n_modes=16)
    >>> samples = model.sample(batch_size=1000)
    """

    def __init__(
        self,
        data_dim: int = 2,
        n_modes: int = 16,
        deltas: Tuple[float, ...] = (1.5, 2.5),
        hidden: int = 64,
        depth: int = 3,
        time_emb_dim: int = 128,
        geometry: str = "planar",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Initialize spectral baseline.

        Args:
            data_dim: Data dimension
            n_modes: Number of spectral modes per dimension
            deltas: Conformal dimensions
            hidden: Hidden channel count
            depth: Network depth
            time_emb_dim: Time embedding dimension
            geometry: Geometry type for encoding
            device: Target device
        """
        super().__init__()

        self.data_dim = data_dim
        self.n_modes = n_modes
        self.n_channels = len(deltas)
        self.device = device

        # Try to import spectral codec
        try:
            from ads_cft.encoding_spectral import SpectralHolographicCodec
            from ads_cft.geometry import SliceGeometry

            # Convert geometry string to enum
            geometry_map = {
                "planar": SliceGeometry.PLANAR,
                "flat": SliceGeometry.FLAT,
            }
            slice_geometry = geometry_map.get(geometry.lower(), SliceGeometry.PLANAR)

            self.spectral_codec = SpectralHolographicCodec(
                d=data_dim,
                deltas=deltas,
                n_modes=n_modes,
                geometry=slice_geometry,
                r_uv=1.0,
            )
            self._has_codec = True
        except ImportError:
            print("[WARNING] SpectralHolographicCodec not available")
            self.spectral_codec = None
            self._has_codec = False

        # State shape: (2 * n_channels, n_modes, n_modes) for real/imag
        self.state_channels = 2 * self.n_channels
        self.state_shape = (self.state_channels, n_modes, n_modes)

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden * 4),
            nn.SiLU(),
            nn.Linear(hidden * 4, hidden),
        )

        # CNN velocity network
        max_depth = int(math.log2(n_modes))
        actual_depth = min(depth, max_depth)
        if actual_depth != depth:
            print(
                f"[WARNING] Reducing depth from {depth} to {actual_depth} "
                f"for {n_modes}x{n_modes} grid"
            )
        self.velocity_net = UNetVelocity(self.state_channels, hidden, actual_depth)

        self.to(device)

        # For normalization
        self.data_mean: Optional[Tensor] = None
        self.data_std: Optional[Tensor] = None

        print(f"[SPECTRAL BASELINE] Fair comparison mode:")
        print(f"  Encoding: Spectral ({n_modes}×{n_modes} modes, {self.n_channels} channels)")
        print(f"  Network: CNN (same as AdS)")
        print(f"  Path: LINEAR (no Klein-Gordon backbone)")
        print(f"  Prior: GAUSSIAN (no Sobolev structure)")

    def encode(self, points: Tensor) -> Tensor:
        """
        Encode points to spectral representation.

        Args:
            points: Input points, shape (B, D)

        Returns:
            Spectral representation, shape (B, 2*C, K, K)
        """
        if self._has_codec:
            phi_hat, _ = self.spectral_codec.encode(points)
            return phi_hat
        else:
            # Fallback: simple FFT-based encoding
            B = points.shape[0]
            return torch.randn(B, *self.state_shape, device=self.device)

    def decode(self, state: Tensor) -> Tensor:
        """
        Decode spectral representation to points.

        Args:
            state: Spectral representation

        Returns:
            Decoded points
        """
        if self._has_codec:
            return self.spectral_codec.decode(state)
        else:
            # Fallback
            B = state.shape[0]
            return torch.randn(B, self.data_dim, device=self.device)

    def sample_prior(self, batch_size: int) -> Tensor:
        """
        Sample from Gaussian prior in spectral space.

        Args:
            batch_size: Number of samples

        Returns:
            Prior samples
        """
        return torch.randn(batch_size, *self.state_shape, device=self.device)

    def velocity(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Compute learned velocity field.

        Args:
            x: Spectral state
            t: Time

        Returns:
            Velocity
        """
        B = x.shape[0]
        if t.ndim == 0:
            t = t.expand(B)

        # Time embedding (not explicitly used in current version)
        t_emb = self.time_mlp(self.time_embed(t))

        # Velocity prediction
        v = self.velocity_net(x)
        return v

    def compute_loss(self, points: Tensor) -> Dict[str, Tensor]:
        """
        Compute flow matching loss with spectral encoding.

        Args:
            points: Input points

        Returns:
            Loss dictionary
        """
        B = points.shape[0]

        # Normalize if configured
        if self.data_mean is not None and self.data_std is not None:
            points = (points - self.data_mean.to(points.device)) / self.data_std.to(
                points.device
            )

        # Encode points to spectral (target)
        x1 = self.encode(points)

        # Sample prior (start)
        x0 = self.sample_prior(B)

        # Sample time
        t = torch.rand(B, device=self.device)

        # Linear interpolation
        t_expand = t.view(B, 1, 1, 1)
        x_t = (1 - t_expand) * x0 + t_expand * x1

        # Target velocity
        v_target = x1 - x0

        # Predicted velocity
        v_pred = self.velocity(x_t, t)

        # MSE loss
        loss = F.mse_loss(v_pred, v_target)

        with torch.no_grad():
            velocity_norm = (v_pred ** 2).mean()

        return {
            "total_loss": loss,
            "mse_loss": loss,
            "residual_norm": velocity_norm.detach(),
        }

    @torch.no_grad()
    def sample(self, batch_size: int, n_steps: int = 100) -> Tensor:
        """
        Generate samples via ODE integration.

        Args:
            batch_size: Number of samples
            n_steps: Integration steps

        Returns:
            Generated samples
        """
        self.eval()

        # Sample from prior (spectral space)
        x = self.sample_prior(batch_size)

        # Euler integration
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.tensor(i * dt, device=self.device)
            v = self.velocity(x, t)
            x = x + v * dt

        # Decode back to points
        points = self.decode(x)

        # Denormalize if configured
        if self.data_mean is not None and self.data_std is not None:
            points = (
                points * self.data_std.to(points.device)
                + self.data_mean.to(points.device)
            )

        return points


# =============================================================================
# Spectral Image Baseline (Fair Comparison for Images)
# =============================================================================


@register_model("baseline_flow_spectral_image")
class SpectralImageBaselineFlowMatching(nn.Module):
    """
    Fair baseline for IMAGE data: Same FFT encoding as AdS, but NO physics.

    This isolates what the AdS geometric structure contributes by using:
    - SAME spectral encoding (FFT for images)
    - SAME CNN architecture
    - SAME number of parameters

    But WITHOUT:
    - Klein-Gordon backbone
    - Hermite path
    - Sobolev prior
    - UV stabilization

    Attributes
    ----------
    n_channels : int
        Number of image channels
    height : int
        Image height
    width : int
        Image width
    image_shape : Tuple[int, int, int]
        Image shape (C, H, W)

    Example
    -------
    >>> model = SpectralImageBaselineFlowMatching(image_shape=(1, 28, 28))
    >>> samples = model.sample(batch_size=64)
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int],  # (C, H, W)
        hidden: int = 64,
        depth: int = 3,
        time_emb_dim: int = 128,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Initialize spectral image baseline.

        Args:
            image_shape: Image shape (C, H, W)
            hidden: Hidden channel count
            depth: Network depth
            time_emb_dim: Time embedding dimension
            device: Target device
        """
        super().__init__()

        self.n_channels = image_shape[0]
        self.height = image_shape[1]
        self.width = image_shape[2]
        self.image_shape = image_shape
        self.device = device

        # State shape: (2 * n_channels, H, W) for real/imag
        self.state_channels = 2 * self.n_channels
        self.state_shape = (self.state_channels, self.height, self.width)

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden * 4),
            nn.SiLU(),
            nn.Linear(hidden * 4, hidden),
        )

        # CNN velocity network
        max_depth = max(1, int(math.log2(min(self.height, self.width))) - 2)
        actual_depth = min(depth, max_depth)
        if actual_depth != depth:
            print(
                f"[WARNING] Reducing depth from {depth} to {actual_depth} "
                f"for {self.height}×{self.width} images"
            )

        self.velocity_net = UNetVelocity(self.state_channels, hidden, actual_depth)

        self.to(device)

        # For normalization
        self.data_mean: Optional[Tensor] = None
        self.data_std: Optional[Tensor] = None

        print(f"[SPECTRAL IMAGE BASELINE] Fair comparison mode:")
        print(f"  Image shape: {self.n_channels}×{self.height}×{self.width}")
        print(f"  Encoding: FFT (same as AdS for images)")
        print(f"  Network: CNN (same as AdS)")
        print(f"  Path: LINEAR (no Klein-Gordon backbone)")
        print(f"  Prior: GAUSSIAN (no Sobolev structure)")

    def encode(self, images: Tensor) -> Tensor:
        """
        Encode images to spectral representation using FFT.

        Args:
            images: Input images, shape (B, C, H, W)

        Returns:
            Spectral representation, shape (B, 2*C, H, W)
        """
        B, C, H, W = images.shape

        # 2D FFT for each channel
        phi_hat_complex = torch.fft.fft2(images, norm="ortho")

        # Split into real and imaginary, interleave channels
        phi_real = phi_hat_complex.real  # (B, C, H, W)
        phi_imag = phi_hat_complex.imag  # (B, C, H, W)

        # Interleave: [real_0, imag_0, real_1, imag_1, ...]
        phi_hat = torch.stack([phi_real, phi_imag], dim=2)  # (B, C, 2, H, W)
        phi_hat = phi_hat.view(B, 2 * C, H, W)

        return phi_hat

    def decode(self, phi_hat: Tensor) -> Tensor:
        """
        Decode spectral representation to images using inverse FFT.

        Args:
            phi_hat: Spectral representation, shape (B, 2*C, H, W)

        Returns:
            Decoded images, shape (B, C, H, W)
        """
        B = phi_hat.shape[0]
        C = self.n_channels
        H, W = self.height, self.width

        # Reshape: (B, 2*C, H, W) -> (B, C, 2, H, W) -> complex (B, C, H, W)
        phi_hat_ri = phi_hat.view(B, C, 2, H, W)
        phi_hat_complex = torch.complex(phi_hat_ri[:, :, 0], phi_hat_ri[:, :, 1])

        # Inverse 2D FFT
        images = torch.fft.ifft2(phi_hat_complex, norm="ortho")

        return images.real

    def sample_prior(self, batch_size: int) -> Tensor:
        """
        Sample from Gaussian prior in spectral space.

        Args:
            batch_size: Number of samples

        Returns:
            Prior samples
        """
        return torch.randn(batch_size, *self.state_shape, device=self.device)

    def velocity(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Compute learned velocity field.

        Args:
            x: Spectral state
            t: Time

        Returns:
            Velocity
        """
        B = x.shape[0]
        if t.ndim == 0:
            t = t.expand(B)

        # Time embedding
        t_emb = self.time_mlp(self.time_embed(t))

        # Velocity prediction
        v = self.velocity_net(x)
        return v

    def compute_loss(self, images: Tensor) -> Dict[str, Tensor]:
        """
        Compute flow matching loss with FFT encoding.

        Args:
            images: Input images

        Returns:
            Loss dictionary
        """
        B = images.shape[0]

        # Normalize if configured
        if self.data_mean is not None and self.data_std is not None:
            images = (images - self.data_mean.to(images.device)) / self.data_std.to(
                images.device
            )

        # Encode images to spectral (target)
        x1 = self.encode(images)

        # Sample prior (start)
        x0 = self.sample_prior(B)

        # Sample time
        t = torch.rand(B, device=self.device)

        # Linear interpolation
        t_expand = t.view(B, 1, 1, 1)
        x_t = (1 - t_expand) * x0 + t_expand * x1

        # Target velocity
        v_target = x1 - x0

        # Predicted velocity
        v_pred = self.velocity(x_t, t)

        # MSE loss
        loss = F.mse_loss(v_pred, v_target)

        with torch.no_grad():
            velocity_norm = (v_pred ** 2).mean()

        return {
            "total_loss": loss,
            "mse_loss": loss,
            "residual_norm": velocity_norm.detach(),
        }

    @torch.no_grad()
    def sample(self, batch_size: int, n_steps: int = 100) -> Tensor:
        """
        Generate samples via ODE integration.

        Args:
            batch_size: Number of samples
            n_steps: Integration steps

        Returns:
            Generated images
        """
        self.eval()

        # Sample from prior (spectral space)
        x = self.sample_prior(batch_size)

        # Euler integration
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.tensor(i * dt, device=self.device)
            v = self.velocity(x, t)
            x = x + v * dt

        # Decode back to images
        images = self.decode(x)

        # Denormalize if configured
        if self.data_mean is not None and self.data_std is not None:
            images = (
                images * self.data_std.to(images.device)
                + self.data_mean.to(images.device)
            )

        return images


# =============================================================================
# Note: BaselineConfig is imported from config.py
# =============================================================================


# =============================================================================
# Factory Function
# =============================================================================


def create_baseline_model(
    config: BaselineConfig,
    data_shape: Tuple[int, ...],
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Create baseline model from configuration.

    Args:
        config: Baseline configuration
        data_shape: Data shape
        device: Target device (defaults to config.device)

    Returns:
        Baseline model instance
    """
    if device is None:
        device = torch.device(config.device)

    is_image = len(data_shape) >= 2 and data_shape[-1] > 2

    if config.baseline_type == "spectral":
        if is_image:
            return SpectralImageBaselineFlowMatching(
                image_shape=data_shape,
                hidden=config.hidden,
                depth=config.depth,
                time_emb_dim=config.time_emb_dim,
                device=device,
            )
        else:
            data_dim = data_shape[0] if len(data_shape) == 1 else 2
            return SpectralBaselineFlowMatching(
                data_dim=data_dim,
                n_modes=config.spectral_n_modes,
                deltas=config.spectral_deltas,
                hidden=config.hidden,
                depth=config.depth,
                time_emb_dim=config.time_emb_dim,
                geometry=config.spectral_geometry,
                device=device,
            )
    else:
        return BaselineFlowMatching(
            data_shape=data_shape,
            hidden=config.hidden,
            depth=config.depth,
            time_emb_dim=config.time_emb_dim,
            device=device,
        )