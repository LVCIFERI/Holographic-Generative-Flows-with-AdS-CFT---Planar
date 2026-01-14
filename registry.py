"""
registry.py

Central Registry System for AdS/CFT Flow Matching Framework
============================================================

This module provides a unified, extensible registration system for all pluggable
components in the framework. It enables:

1. **Easy Extension**: Add new geometries, Laplacians, encoders, etc. with decorators
2. **Runtime Discovery**: List all available components programmatically
3. **Unified Factory Interface**: Create any component by name with consistent API
4. **Type Safety**: Strong typing with Protocol classes for component interfaces

Design Pattern
--------------
The registry uses the "Registry" pattern combined with decorator-based registration:

    @register_geometry("my_geometry")
    class MyGeometry(AdSGeometry):
        ...

This automatically adds the class to the global registry, making it available
for instantiation via the factory function:

    geometry = create_geometry("my_geometry", d=2, ...)

Component Types
---------------
The following component types are supported:

    - geometry: AdS slice geometries (planar, flat, planar_hsv)
    - laplacian: Slice Laplacian operators (diagonal)
    - encoder: Point-to-field encoders (spectral, holographic, image)
    - dataset: Data samplers (toy distributions, image datasets)
    - solver: ODE integration methods (euler, heun, rk4)
    - baseline: Baseline comparison models (MLP, spectral)

Usage Examples
--------------
Registering a new geometry:

    >>> from ads_cft.registry import register_geometry
    >>> from ads_cft.geometry import AdSGeometry
    >>> 
    >>> @register_geometry("torus")
    >>> class TorusAdS(AdSGeometry):
    ...     def f(self, r):
    ...         return torch.cosh(r)
    ...     # ... other methods

Creating components:

    >>> from ads_cft.registry import create_geometry, create_laplacian
    >>> 
    >>> geometry = create_geometry("planar", d=2, r_min=0.0, r_max=8.0)
    >>> laplacian = create_laplacian("diagonal", geometry=geometry, n_modes=16)

Listing available components:

    >>> from ads_cft.registry import list_available, print_registry_status
    >>> 
    >>> print(list_available("geometry"))
    ['planar', 'flat', 'planar_hsv']
    >>> 
    >>> print_registry_status()  # Pretty-print all registries

Thread Safety
-------------
The registry is NOT thread-safe for registration (which typically happens at
module import time). Component creation via factories IS thread-safe.

Document References
-------------------
- Section 2: Geometry definitions (planar)
- Section 3: Klein-Gordon backbone and Laplacian operators
- Section 5: Field encoding and UV lift/readout
- Algorithm 2: ODE integration for sampling
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    runtime_checkable,
)

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ads_cft.config import (
        ExperimentConfig,
        FlowModelConfig,
        GeometryConfig,
    )

logger = logging.getLogger(__name__)

# Type variable for generic registration
T = TypeVar("T")

# =============================================================================
# Protocol Definitions (Component Interfaces)
# =============================================================================


@runtime_checkable
class GeometryProtocol(Protocol):
    """Protocol for AdS geometry implementations."""

    @property
    def d(self) -> int:
        """Boundary dimension."""
        ...

    def f(self, r: torch.Tensor) -> torch.Tensor:
        """Warp factor f(r) for single-warp metrics."""
        ...

    def log_f_prime(self, r: torch.Tensor) -> torch.Tensor:
        """Return (f'/f)(r)."""
        ...

    def slice_weight(self, r: torch.Tensor) -> torch.Tensor:
        """Volume weight ω(r) such that dvol_{g_r} = ω(r) dvol_ĝ."""
        ...

    def kappa(self, r: torch.Tensor) -> torch.Tensor:
        """Radial friction κ(r) in the KG equation."""
        ...


@runtime_checkable
class LaplacianProtocol(Protocol):
    """Protocol for slice Laplacian implementations."""

    def apply_minus_laplacian(self, phi: torch.Tensor) -> torch.Tensor:
        """Apply (-Δ_ĝ) to field φ."""
        ...

    def diag_eigs(
        self, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        """Return diagonal eigenvalues if available."""
        ...


@runtime_checkable
class EncoderProtocol(Protocol):
    """Protocol for point-to-field encoder implementations."""

    def encode(self, points: torch.Tensor) -> torch.Tensor:
        """Encode points to field representation."""
        ...

    def decode(self, field: torch.Tensor) -> torch.Tensor:
        """Decode field back to points."""
        ...


@runtime_checkable
class SolverProtocol(Protocol):
    """Protocol for ODE solver step functions."""

    def __call__(
        self,
        drift_fn: Callable,
        y: Any,
        r: torch.Tensor,
        h: float,
    ) -> Any:
        """Perform one integration step."""
        ...


# =============================================================================
# Global Registries
# =============================================================================

# Each registry maps string names to classes/functions
GEOMETRY_REGISTRY: Dict[str, Type] = {}
LAPLACIAN_REGISTRY: Dict[str, Type] = {}
ENCODER_REGISTRY: Dict[str, Type] = {}
DATASET_REGISTRY: Dict[str, Callable] = {}
SOLVER_REGISTRY: Dict[str, Callable] = {}
BASELINE_REGISTRY: Dict[str, Type] = {}
NETWORK_REGISTRY: Dict[str, Type] = {}
MODEL_REGISTRY: Dict[str, Type] = {}

# Mapping from component type to registry
_REGISTRY_MAP: Dict[str, Dict] = {
    "geometry": GEOMETRY_REGISTRY,
    "laplacian": LAPLACIAN_REGISTRY,
    "encoder": ENCODER_REGISTRY,
    "dataset": DATASET_REGISTRY,
    "solver": SOLVER_REGISTRY,
    "baseline": BASELINE_REGISTRY,
    "network": NETWORK_REGISTRY,
    "model": MODEL_REGISTRY,
}


# =============================================================================
# Registration Decorators
# =============================================================================


def register_geometry(name: str) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register a geometry class.

    Args:
        name: Unique identifier for the geometry (e.g., "planar", "flat")

    Returns:
        Decorator function that registers the class

    Example:
        >>> @register_geometry("torus")
        >>> class TorusAdS(AdSGeometry):
        ...     pass
    """

    def decorator(cls: Type[T]) -> Type[T]:
        if name in GEOMETRY_REGISTRY:
            logger.warning(f"Overwriting geometry registration: {name}")
        GEOMETRY_REGISTRY[name] = cls
        logger.debug(f"Registered geometry: {name} -> {cls.__name__}")
        return cls

    return decorator


def register_laplacian(name: str) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register a Laplacian operator class.

    Args:
        name: Unique identifier (e.g., "diagonal", "planar_spectral")

    Returns:
        Decorator function that registers the class

    Example:
        >>> @register_laplacian("my_laplacian")
        >>> class MyLaplacian(SliceLaplacian):
        ...     pass
    """

    def decorator(cls: Type[T]) -> Type[T]:
        if name in LAPLACIAN_REGISTRY:
            logger.warning(f"Overwriting laplacian registration: {name}")
        LAPLACIAN_REGISTRY[name] = cls
        logger.debug(f"Registered laplacian: {name} -> {cls.__name__}")
        return cls

    return decorator


def register_encoder(name: str) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register an encoder class.

    Args:
        name: Unique identifier (e.g., "spectral", "holographic", "image")

    Returns:
        Decorator function that registers the class

    Example:
        >>> @register_encoder("my_encoder")
        >>> class MyEncoder(nn.Module):
        ...     pass
    """

    def decorator(cls: Type[T]) -> Type[T]:
        if name in ENCODER_REGISTRY:
            logger.warning(f"Overwriting encoder registration: {name}")
        ENCODER_REGISTRY[name] = cls
        logger.debug(f"Registered encoder: {name} -> {cls.__name__}")
        return cls

    return decorator


def register_dataset(name: str) -> Callable[[Callable], Callable]:
    """
    Decorator to register a dataset factory function.

    Args:
        name: Unique identifier (e.g., "checkerboard", "mnist")

    Returns:
        Decorator function that registers the factory

    Example:
        >>> @register_dataset("my_dataset")
        >>> def create_my_dataset(n_samples: int, **kwargs):
        ...     return MyDataset(n_samples, **kwargs)
    """

    def decorator(func: Callable) -> Callable:
        if name in DATASET_REGISTRY:
            logger.warning(f"Overwriting dataset registration: {name}")
        DATASET_REGISTRY[name] = func
        logger.debug(f"Registered dataset: {name} -> {func.__name__}")
        return func

    return decorator


def register_solver(name: str) -> Callable[[Callable], Callable]:
    """
    Decorator to register an ODE solver function.

    Args:
        name: Unique identifier (e.g., "euler", "rk4", "heun")

    Returns:
        Decorator function that registers the solver

    Example:
        >>> @register_solver("my_solver")
        >>> def my_solver_step(drift_fn, y, r, h):
        ...     return y + h * drift_fn(y, r)
    """

    def decorator(func: Callable) -> Callable:
        if name in SOLVER_REGISTRY:
            logger.warning(f"Overwriting solver registration: {name}")
        SOLVER_REGISTRY[name] = func
        logger.debug(f"Registered solver: {name} -> {func.__name__}")
        return func

    return decorator


def register_baseline(name: str) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register a baseline model class.

    Args:
        name: Unique identifier (e.g., "mlp", "spectral", "unet")

    Returns:
        Decorator function that registers the class

    Example:
        >>> @register_baseline("my_baseline")
        >>> class MyBaseline(nn.Module):
        ...     pass
    """

    def decorator(cls: Type[T]) -> Type[T]:
        if name in BASELINE_REGISTRY:
            logger.warning(f"Overwriting baseline registration: {name}")
        BASELINE_REGISTRY[name] = cls
        logger.debug(f"Registered baseline: {name} -> {cls.__name__}")
        return cls

    return decorator


def register_network(name: str) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register a neural network class.

    Args:
        name: Unique identifier for the network (e.g., "residual_mlp", "residual_cnn")

    Returns:
        Decorator function that registers the network class

    Example:
        >>> @register_network("my_net")
        >>> class MyNetwork(nn.Module):
        ...     def forward(self, x, t):
        ...         return x
    """

    def decorator(cls: Type[T]) -> Type[T]:
        if name in NETWORK_REGISTRY:
            logger.warning(f"Overwriting network registration: {name}")
        NETWORK_REGISTRY[name] = cls
        logger.debug(f"Registered network: {name} -> {cls.__name__}")
        return cls

    return decorator


def register_model(name: str) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register a model class.

    Args:
        name: Unique identifier for the model (e.g., "uv_flow_matching")

    Returns:
        Decorator function that registers the model class

    Example:
        >>> @register_model("my_model")
        >>> class MyModel(nn.Module):
        ...     def forward(self, x):
        ...         return x
    """

    def decorator(cls: Type[T]) -> Type[T]:
        if name in MODEL_REGISTRY:
            logger.warning(f"Overwriting model registration: {name}")
        MODEL_REGISTRY[name] = cls
        logger.debug(f"Registered model: {name} -> {cls.__name__}")
        return cls

    return decorator


# =============================================================================
# Factory Functions
# =============================================================================


def create_geometry(name: str, **kwargs) -> "GeometryProtocol":
    """
    Create a geometry instance by name.

    Args:
        name: Registered geometry name
        **kwargs: Arguments passed to geometry constructor

    Returns:
        Geometry instance

    Raises:
        ValueError: If geometry name is not registered

    Example:
        >>> geometry = create_geometry("planar", d=2, r_min=0.0, r_max=8.0)
    """
    if name not in GEOMETRY_REGISTRY:
        available = list(GEOMETRY_REGISTRY.keys())
        raise ValueError(
            f"Unknown geometry '{name}'. Available: {available}. "
            f"Use @register_geometry('{name}') to register a new geometry."
        )
    return GEOMETRY_REGISTRY[name](**kwargs)


def create_laplacian(name: str, **kwargs) -> "LaplacianProtocol":
    """
    Create a Laplacian operator instance by name.

    Args:
        name: Registered laplacian name
        **kwargs: Arguments passed to laplacian constructor

    Returns:
        Laplacian instance

    Raises:
        ValueError: If laplacian name is not registered

    Example:
        >>> laplacian = create_laplacian("diagonal", geometry_type="planar", n_modes=16)
    """
    if name not in LAPLACIAN_REGISTRY:
        available = list(LAPLACIAN_REGISTRY.keys())
        raise ValueError(
            f"Unknown laplacian '{name}'. Available: {available}. "
            f"Use @register_laplacian('{name}') to register a new laplacian."
        )
    return LAPLACIAN_REGISTRY[name](**kwargs)


def create_encoder(name: str, **kwargs) -> "EncoderProtocol":
    """
    Create an encoder instance by name.

    Args:
        name: Registered encoder name
        **kwargs: Arguments passed to encoder constructor

    Returns:
        Encoder instance

    Raises:
        ValueError: If encoder name is not registered

    Example:
        >>> encoder = create_encoder("spectral", n_modes=16, deltas=(1.5, 2.5))
    """
    if name not in ENCODER_REGISTRY:
        available = list(ENCODER_REGISTRY.keys())
        raise ValueError(
            f"Unknown encoder '{name}'. Available: {available}. "
            f"Use @register_encoder('{name}') to register a new encoder."
        )
    return ENCODER_REGISTRY[name](**kwargs)


def create_dataset(name: str, **kwargs) -> Any:
    """
    Create a dataset by name using the registered factory.

    Args:
        name: Registered dataset name
        **kwargs: Arguments passed to dataset factory

    Returns:
        Dataset instance (type depends on the specific dataset)

    Raises:
        ValueError: If dataset name is not registered

    Example:
        >>> dataset = create_dataset("checkerboard", n_samples=10000)
    """
    if name not in DATASET_REGISTRY:
        available = list(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {available}. "
            f"Use @register_dataset('{name}') to register a new dataset."
        )
    return DATASET_REGISTRY[name](**kwargs)


def create_solver(name: str) -> Callable:
    """
    Get a solver step function by name.

    Args:
        name: Registered solver name

    Returns:
        Solver step function

    Raises:
        ValueError: If solver name is not registered

    Example:
        >>> solver = create_solver("rk4")
        >>> y_next = solver(drift_fn, y, r, h)
    """
    if name not in SOLVER_REGISTRY:
        available = list(SOLVER_REGISTRY.keys())
        raise ValueError(
            f"Unknown solver '{name}'. Available: {available}. "
            f"Use @register_solver('{name}') to register a new solver."
        )
    return SOLVER_REGISTRY[name]


def create_baseline(name: str, **kwargs) -> nn.Module:
    """
    Create a baseline model instance by name.

    Args:
        name: Registered baseline name
        **kwargs: Arguments passed to baseline constructor

    Returns:
        Baseline model instance

    Raises:
        ValueError: If baseline name is not registered

    Example:
        >>> baseline = create_baseline("mlp", hidden=256, depth=4)
    """
    if name not in BASELINE_REGISTRY:
        available = list(BASELINE_REGISTRY.keys())
        raise ValueError(
            f"Unknown baseline '{name}'. Available: {available}. "
            f"Use @register_baseline('{name}') to register a new baseline."
        )
    return BASELINE_REGISTRY[name](**kwargs)


def create_model(config: "ExperimentConfig") -> nn.Module:
    """
    Create a complete model from experiment configuration.

    This is the main entry point that orchestrates creation of all components:
    geometry, encoder, laplacian, and neural networks.

    Args:
        config: Complete experiment configuration

    Returns:
        Configured model (UVStabilizedFlowMatchingModel or baseline)

    Note:
        This function is implemented in model.py to avoid circular imports.
        It is re-exported here for API convenience.

    Example:
        >>> config = ExperimentConfig(dataset="checkerboard", use_spectral_encoding=True)
        >>> model = create_model(config)
    """
    # Defer import to avoid circular dependency
    from ads_cft.model import create_model as _create_model

    return _create_model(config)


# =============================================================================
# Discovery Functions
# =============================================================================


def list_available(component_type: str) -> List[str]:
    """
    List all registered components of a given type.

    Args:
        component_type: One of "geometry", "laplacian", "encoder", "dataset",
                       "solver", "baseline"

    Returns:
        List of registered component names

    Raises:
        ValueError: If component_type is not recognized

    Example:
        >>> print(list_available("geometry"))
        ['planar', 'flat', 'planar_hsv']
    """
    if component_type not in _REGISTRY_MAP:
        valid_types = list(_REGISTRY_MAP.keys())
        raise ValueError(
            f"Unknown component type '{component_type}'. Valid types: {valid_types}"
        )
    return list(_REGISTRY_MAP[component_type].keys())


def get_registry(component_type: str) -> Dict[str, Any]:
    """
    Get the raw registry dictionary for a component type.

    Args:
        component_type: One of "geometry", "laplacian", "encoder", etc.

    Returns:
        Registry dictionary mapping names to classes/functions

    Note:
        Modifying the returned dictionary will affect the global registry.
        Use with caution.
    """
    if component_type not in _REGISTRY_MAP:
        valid_types = list(_REGISTRY_MAP.keys())
        raise ValueError(
            f"Unknown component type '{component_type}'. Valid types: {valid_types}"
        )
    return _REGISTRY_MAP[component_type]


def is_registered(component_type: str, name: str) -> bool:
    """
    Check if a component is registered.

    Args:
        component_type: Type of component to check
        name: Name to look up

    Returns:
        True if registered, False otherwise
    """
    registry = _REGISTRY_MAP.get(component_type, {})
    return name in registry


def print_registry_status() -> None:
    """
    Print a formatted summary of all registered components.

    Useful for debugging and discovering available options.

    Example output:
        ════════════════════════════════════════════════════════════════
        REGISTERED COMPONENTS
        ════════════════════════════════════════════════════════════════
        geometry   : planar, flat, planar_hsv
        laplacian  : diagonal, planar_spectral
        encoder    : spectral, holographic, image
        dataset    : checkerboard, gaussian_mixture, mnist, ...
        solver     : euler, heun, midpoint, rk4
        baseline   : mlp, spectral, unet
        ════════════════════════════════════════════════════════════════
    """
    width = 68
    print("═" * width)
    print("REGISTERED COMPONENTS")
    print("═" * width)

    for component_type in _REGISTRY_MAP.keys():
        items = list_available(component_type)
        items_str = ", ".join(items) if items else "(none)"
        print(f"{component_type:12}: {items_str}")

    print("═" * width)


def clear_registry(component_type: Optional[str] = None) -> None:
    """
    Clear one or all registries.

    Primarily for testing purposes.

    Args:
        component_type: If provided, clear only this registry.
                       If None, clear all registries.

    Warning:
        This will break any code depending on the cleared registrations!
    """
    if component_type is None:
        for registry in _REGISTRY_MAP.values():
            registry.clear()
        logger.warning("Cleared ALL registries")
    else:
        if component_type not in _REGISTRY_MAP:
            raise ValueError(f"Unknown component type: {component_type}")
        _REGISTRY_MAP[component_type].clear()
        logger.warning(f"Cleared registry: {component_type}")


# =============================================================================
# Validation Utilities
# =============================================================================


def validate_component(
    component_type: str,
    instance: Any,
    raise_on_fail: bool = True,
) -> bool:
    """
    Validate that an instance implements the required protocol.

    Args:
        component_type: Type of component to validate against
        instance: Instance to validate
        raise_on_fail: If True, raise TypeError on validation failure

    Returns:
        True if valid, False otherwise (only if raise_on_fail=False)

    Raises:
        TypeError: If validation fails and raise_on_fail=True
    """
    protocols = {
        "geometry": GeometryProtocol,
        "laplacian": LaplacianProtocol,
        "encoder": EncoderProtocol,
    }

    if component_type not in protocols:
        # No protocol defined for this type, accept anything
        return True

    protocol = protocols[component_type]
    is_valid = isinstance(instance, protocol)

    if not is_valid and raise_on_fail:
        raise TypeError(
            f"Instance {type(instance).__name__} does not implement "
            f"{protocol.__name__} protocol for component type '{component_type}'"
        )

    return is_valid


# =============================================================================
# Module Initialization
# =============================================================================


def _auto_register_builtins() -> None:
    """
    Auto-register built-in components when their modules are imported.

    This is called at module import time to ensure built-in components
    are always available.

    Note:
        Individual modules (geometry.py, laplacian_*.py, etc.) use the
        registration decorators to register themselves when imported.
    """
    # Built-in registrations happen in their respective modules
    # This function exists for potential future use (e.g., lazy loading)
    pass


# Run auto-registration
_auto_register_builtins()