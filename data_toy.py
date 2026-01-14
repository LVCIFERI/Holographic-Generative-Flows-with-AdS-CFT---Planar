"""
data_toy.py

Toy Datasets for Testing UV-Stabilized Generative Flow Matching
================================================================

Document context:
- Section 2: AdS foliations (planar)
- The slice manifold Σ ≅ ℝ^d for planar geometry
- Data lives on the boundary, which is conformally equivalent to the UV slice

Supported Toy Distributions
---------------------------

PLANAR (Σ ≅ ℝ^d):
   - Checkerboard, Gaussian mixture, Swiss roll, moons, circles, pinwheel
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch
from torch.utils.data import Dataset

from ads_cft.config import DatasetType

Tensor = torch.Tensor


# =============================================================================
# Note: DatasetType is imported from config.py to avoid duplication
# =============================================================================


# =============================================================================
# Base Sampler Interface
# =============================================================================


class ToyDataSampler:
    """
    Base class for toy data samplers.

    All samplers implement the `sample` method to generate data points
    and provide `data_dim` and `data_shape` properties.

    Example
    -------
    >>> sampler = CheckerboardSampler()
    >>> samples = sampler.sample(1000)  # (1000, 2)
    """

    def sample(
        self,
        n_samples: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """
        Sample n_samples points.

        Args:
            n_samples: Number of samples to generate
            device: Target device for tensors
            dtype: Data type for tensors
            generator: Random number generator for reproducibility

        Returns:
            Tensor of shape (n_samples, data_dim)
        """
        raise NotImplementedError

    @property
    def data_dim(self) -> int:
        """Dimensionality of the data."""
        raise NotImplementedError

    @property
    def data_shape(self) -> Tuple[int, ...]:
        """Shape of a single sample."""
        return (self.data_dim,)


# =============================================================================
# Planar Datasets (Σ ≅ ℝ^d)
# =============================================================================


class CheckerboardSampler(ToyDataSampler):
    """
    2D checkerboard distribution.

    Standard benchmark for flow-based models. Samples are drawn
    uniformly from alternating tiles of a checkerboard pattern.

    Default range: [-4, 4] × [-4, 4] (scale=8.0)

    Attributes
    ----------
    n_tiles : int
        Number of tiles per dimension
    scale : float
        Total range of the checkerboard
    noise : float
        Optional Gaussian noise to add

    Example
    -------
    >>> sampler = CheckerboardSampler(n_tiles=4, scale=8.0)
    >>> samples = sampler.sample(10000)
    """

    def __init__(
        self,
        n_tiles: int = 4,
        scale: float = 8.0,
        noise: float = 0.0,
    ) -> None:
        """
        Initialize checkerboard sampler.

        Args:
            n_tiles: Number of tiles per dimension (default: 4)
            scale: Total range of the checkerboard (default: 8.0)
            noise: Standard deviation of Gaussian noise (default: 0.0)
        """
        self.n_tiles = n_tiles
        self.scale = scale
        self.noise = noise

    @property
    def data_dim(self) -> int:
        return 2

    def sample(
        self,
        n_samples: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        device = device or torch.device("cpu")

        # Sample uniform in [0, n_tiles] × [0, n_tiles]
        x = (
            torch.rand(n_samples, 2, device=device, dtype=dtype, generator=generator)
            * self.n_tiles
        )

        # Get tile indices
        ix = x[:, 0].long()
        iy = x[:, 1].long()

        # Checkerboard pattern: keep only tiles where (ix + iy) % 2 == 0
        mask = (ix + iy) % 2 == 0

        # Rejection sampling
        n_accepted = mask.sum().item()
        while n_accepted < n_samples:
            x_new = (
                torch.rand(
                    n_samples, 2, device=device, dtype=dtype, generator=generator
                )
                * self.n_tiles
            )
            ix_new = x_new[:, 0].long()
            iy_new = x_new[:, 1].long()
            mask_new = (ix_new + iy_new) % 2 == 0

            n_needed = n_samples - n_accepted
            n_take = min(mask_new.sum().item(), n_needed)

            if n_take > 0:
                x = torch.cat(
                    [x[mask][:n_accepted], x_new[mask_new][:n_take]], dim=0
                )
                n_accepted = x.shape[0]
                mask = torch.ones(n_accepted, dtype=torch.bool, device=device)

        x = x[:n_samples]

        # Add noise
        x = x + self.noise * torch.randn(
            x.shape, device=x.device, dtype=x.dtype, generator=generator
        )

        # Center and scale
        x = (x - self.n_tiles / 2) * (self.scale / self.n_tiles)

        return x


class GaussianMixtureSampler(ToyDataSampler):
    """
    Mixture of Gaussians in ℝ^d.

    Default: 8 components arranged in a circle (2D) or on sphere surface (higher d).

    Attributes
    ----------
    dim : int
        Data dimension
    n_components : int
        Number of mixture components
    radius : float
        Radius of the circle/sphere on which centers are placed
    std : float
        Standard deviation of each component
    centers : Tensor
        Component centers (n_components, dim)
    weights : Tensor
        Component weights (n_components,)

    Example
    -------
    >>> sampler = GaussianMixtureSampler(dim=2, n_components=8)
    >>> samples = sampler.sample(10000)
    """

    def __init__(
        self,
        dim: int = 2,
        n_components: int = 8,
        radius: float = 3.0,
        std: float = 0.3,
        weights: Optional[Tensor] = None,
    ) -> None:
        """
        Initialize Gaussian mixture sampler.

        Args:
            dim: Data dimension (default: 2)
            n_components: Number of mixture components (default: 8)
            radius: Radius of center circle/sphere (default: 3.0)
            std: Standard deviation of each component (default: 0.3)
            weights: Component weights (default: uniform)
        """
        self.dim = dim
        self.n_components = n_components
        self.radius = radius
        self.std = std

        # Generate component centers
        if dim == 2:
            # Arrange on circle
            angles = torch.linspace(0, 2 * math.pi, n_components + 1)[:-1]
            self.centers = torch.stack(
                [
                    radius * torch.cos(angles),
                    radius * torch.sin(angles),
                ],
                dim=1,
            )
        else:
            # Random centers on sphere surface
            centers = torch.randn(n_components, dim)
            centers = centers / centers.norm(dim=1, keepdim=True) * radius
            self.centers = centers

        # Component weights
        if weights is not None:
            self.weights = weights / weights.sum()
        else:
            self.weights = torch.ones(n_components) / n_components

    @property
    def data_dim(self) -> int:
        return self.dim

    def sample(
        self,
        n_samples: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        device = device or torch.device("cpu")
        centers = self.centers.to(device=device, dtype=dtype)
        weights = self.weights.to(device=device, dtype=dtype)

        # Sample component indices
        indices = torch.multinomial(
            weights, n_samples, replacement=True, generator=generator
        )

        # Sample from selected components
        samples = centers[indices] + self.std * torch.randn(
            n_samples, self.dim, device=device, dtype=dtype, generator=generator
        )

        return samples


class SwissRollSampler(ToyDataSampler):
    """
    Swiss roll distribution (2D manifold in 3D, projected to 2D).

    Attributes
    ----------
    noise : float
        Gaussian noise standard deviation
    n_turns : float
        Number of turns in the roll
    scale : float
        Overall scale factor

    Example
    -------
    >>> sampler = SwissRollSampler()
    >>> samples = sampler.sample(10000)
    """

    def __init__(
        self,
        noise: float = 0.0,
        n_turns: float = 1.5,
        scale: float = 1.0,
    ) -> None:
        """
        Initialize Swiss roll sampler.

        Args:
            noise: Gaussian noise standard deviation (default: 0.0)
            n_turns: Number of spiral turns (default: 1.5)
            scale: Overall scale factor (default: 1.0)
        """
        self.noise = noise
        self.n_turns = n_turns
        self.scale = scale

    @property
    def data_dim(self) -> int:
        return 2

    def sample(
        self,
        n_samples: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        device = device or torch.device("cpu")

        # Parameter along the roll
        t = torch.rand(n_samples, device=device, dtype=dtype, generator=generator)
        t = (3 * math.pi / 2) * (1 + 2 * t)  # t ∈ [3π/2, 9π/2]

        # Swiss roll coordinates (project to 2D)
        x = t * torch.cos(t * self.n_turns)
        y = t * torch.sin(t * self.n_turns)

        samples = torch.stack([x, y], dim=1) * self.scale / 10.0

        # Add noise
        samples = samples + self.noise * torch.randn(
            samples.shape, device=samples.device, dtype=samples.dtype, generator=generator
        )

        return samples


class TwoMoonsSampler(ToyDataSampler):
    """
    Two interleaving half-moons distribution.

    Attributes
    ----------
    noise : float
        Gaussian noise standard deviation
    scale : float
        Overall scale factor

    Example
    -------
    >>> sampler = TwoMoonsSampler()
    >>> samples = sampler.sample(10000)
    """

    def __init__(
        self,
        noise: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        """
        Initialize two moons sampler.

        Args:
            noise: Gaussian noise standard deviation (default: 0.0)
            scale: Overall scale factor (default: 1.0)
        """
        self.noise = noise
        self.scale = scale

    @property
    def data_dim(self) -> int:
        return 2

    def sample(
        self,
        n_samples: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        device = device or torch.device("cpu")

        n_per_moon = n_samples // 2
        n_extra = n_samples - 2 * n_per_moon

        # First moon (upper)
        theta1 = (
            torch.rand(
                n_per_moon + n_extra, device=device, dtype=dtype, generator=generator
            )
            * math.pi
        )
        x1 = torch.cos(theta1)
        y1 = torch.sin(theta1)

        # Second moon (lower, shifted)
        theta2 = (
            torch.rand(n_per_moon, device=device, dtype=dtype, generator=generator)
            * math.pi
        )
        x2 = 1.0 - torch.cos(theta2)
        y2 = 0.5 - torch.sin(theta2)

        # Combine
        x = torch.cat([x1, x2])
        y = torch.cat([y1, y2])
        samples = torch.stack([x, y], dim=1) * self.scale

        # Add noise
        samples = samples + self.noise * torch.randn(
            samples.shape, device=samples.device, dtype=samples.dtype, generator=generator
        )

        return samples


class ConcentricCirclesSampler(ToyDataSampler):
    """
    Concentric circles distribution.

    Attributes
    ----------
    n_circles : int
        Number of concentric circles
    noise : float
        Gaussian noise standard deviation
    scale : float
        Radius of outermost circle

    Example
    -------
    >>> sampler = ConcentricCirclesSampler(n_circles=3)
    >>> samples = sampler.sample(10000)
    """

    def __init__(
        self,
        n_circles: int = 3,
        noise: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        """
        Initialize concentric circles sampler.

        Args:
            n_circles: Number of circles (default: 3)
            noise: Gaussian noise standard deviation (default: 0.0)
            scale: Maximum radius (default: 1.0)
        """
        self.n_circles = n_circles
        self.noise = noise
        self.scale = scale

    @property
    def data_dim(self) -> int:
        return 2

    def sample(
        self,
        n_samples: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        device = device or torch.device("cpu")

        # Assign samples to circles
        circle_idx = torch.randint(
            0, self.n_circles, (n_samples,), device=device, generator=generator
        )
        radii = (circle_idx.float() + 1) / self.n_circles * self.scale

        # Sample angles
        angles = (
            torch.rand(n_samples, device=device, dtype=dtype, generator=generator)
            * 2
            * math.pi
        )

        # Generate points
        x = radii * torch.cos(angles)
        y = radii * torch.sin(angles)
        samples = torch.stack([x, y], dim=1)

        # Add noise
        samples = samples + self.noise * torch.randn(
            samples.shape, device=samples.device, dtype=samples.dtype, generator=generator
        )

        return samples


class PinwheelSampler(ToyDataSampler):
    """
    Pinwheel distribution with radial arms.

    Attributes
    ----------
    n_arms : int
        Number of spiral arms
    noise : float
        Gaussian noise standard deviation
    scale : float
        Maximum radius

    Example
    -------
    >>> sampler = PinwheelSampler(n_arms=5)
    >>> samples = sampler.sample(10000)
    """

    def __init__(
        self,
        n_arms: int = 5,
        noise: float = 0.0,
        scale: float = 1.0,
    ) -> None:
        """
        Initialize pinwheel sampler.

        Args:
            n_arms: Number of spiral arms (default: 5)
            noise: Gaussian noise standard deviation (default: 0.0)
            scale: Maximum radius (default: 1.0)
        """
        self.n_arms = n_arms
        self.noise = noise
        self.scale = scale

    @property
    def data_dim(self) -> int:
        return 2

    def sample(
        self,
        n_samples: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        device = device or torch.device("cpu")

        # Assign samples to arms
        arm_idx = torch.randint(
            0, self.n_arms, (n_samples,), device=device, generator=generator
        )
        arm_angle = arm_idx.float() * 2 * math.pi / self.n_arms

        # Radial position
        r = (
            torch.rand(n_samples, device=device, dtype=dtype, generator=generator)
            * self.scale
        )

        # Spiral angle offset
        spiral_offset = r * 0.5

        # Total angle
        angle = arm_angle + spiral_offset

        # Generate points
        x = r * torch.cos(angle)
        y = r * torch.sin(angle)
        samples = torch.stack([x, y], dim=1)

        # Add noise
        samples = samples + self.noise * torch.randn(
            samples.shape, device=samples.device, dtype=samples.dtype, generator=generator
        )

        return samples
# =============================================================================
# Factory Function
# =============================================================================


def create_sampler(
    dataset_type: Union[str, DatasetType],
    **kwargs,
) -> ToyDataSampler:
    """
    Create a toy data sampler.

    Args:
        dataset_type: Type of dataset (string or DatasetType enum)
        **kwargs: Additional arguments for the sampler

    Returns:
        ToyDataSampler instance

    Raises:
        ValueError: If dataset_type is unknown

    Example
    -------
    >>> sampler = create_sampler("checkerboard", n_tiles=4)
    >>> sampler = create_sampler(DatasetType.GAUSSIAN_MIXTURE, n_components=8)
    """
    if isinstance(dataset_type, str):
        dataset_type = DatasetType(dataset_type.lower())

    samplers = {
        # Planar
        DatasetType.CHECKERBOARD: CheckerboardSampler,
        DatasetType.GAUSSIAN_MIXTURE: GaussianMixtureSampler,
        DatasetType.SWISS_ROLL: SwissRollSampler,
        DatasetType.TWO_MOONS: TwoMoonsSampler,
        DatasetType.CONCENTRIC_CIRCLES: ConcentricCirclesSampler,
        DatasetType.PINWHEEL: PinwheelSampler,
    }

    if dataset_type not in samplers:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return samplers[dataset_type](**kwargs)


# =============================================================================
# PyTorch Dataset Wrapper
# =============================================================================


class ToyDataset(Dataset):
    """
    PyTorch Dataset wrapper for toy data samplers.

    Pre-generates a fixed set of samples for reproducible training.

    Attributes
    ----------
    data : Tensor
        Pre-generated samples
    data_shape : Tuple[int, ...]
        Shape of a single sample

    Example
    -------
    >>> sampler = CheckerboardSampler()
    >>> dataset = ToyDataset(sampler, n_samples=10000, seed=42)
    >>> dataloader = DataLoader(dataset, batch_size=64)
    """

    def __init__(
        self,
        sampler: ToyDataSampler,
        n_samples: int,
        seed: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initialize toy dataset.

        Args:
            sampler: Toy data sampler
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
            device: Device for the data
            dtype: Data type
        """
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        self.data = sampler.sample(
            n_samples, device=device, dtype=dtype, generator=generator
        )
        self.data_shape = sampler.data_shape

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Tensor:
        return self.data[idx]


def create_toy_dataset(
    dataset_type: Union[str, DatasetType],
    n_samples: int,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> ToyDataset:
    """
    Create a ToyDataset.

    Args:
        dataset_type: Type of dataset
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        device: Device for the data
        dtype: Data type
        **kwargs: Additional arguments for the sampler

    Returns:
        ToyDataset instance

    Example
    -------
    >>> dataset = create_toy_dataset("checkerboard", n_samples=10000, seed=42)
    """
    sampler = create_sampler(dataset_type, **kwargs)
    return ToyDataset(sampler, n_samples, seed=seed, device=device, dtype=dtype)