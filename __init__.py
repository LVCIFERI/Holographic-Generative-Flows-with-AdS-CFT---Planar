"""
AdS/CFT Flow Matching Framework
===============================

A PyTorch implementation of "Literal AdS/CFT-Sliced Generative Modelling via Bulk Flow Matching".

This framework implements holographic generative modeling using the AdS/CFT correspondence,
with UV-stabilized bulk field evolution and spectral encoding on various geometric slicings.

Package Structure
-----------------
Core Components:
    - config: All configuration dataclasses (geometry, model, training, etc.)
    - registry: Extensible component registration system
    - geometry: AdS warped-product geometries (planar, flat, planar_hsv)

Laplacian Operators:
    - laplacian_base: Slice Laplacian base classes and diagonal implementations

Encoding:
    - encoding_base: Encoder protocols and core classes
    - encoding_spectral: Spectral holographic encoders
    - encoding_image: Image spectral encoders

Neural Networks:
    - networks: Neural network components (embeddings, backbones, residual nets)
    - model: Main UVStabilizedFlowMatchingModel
    - baselines: Baseline comparison models

Dynamics:
    - paths: Path interpolation (linear, Hermite)
    - evolution: ODE solvers
    - losses: Loss functions

Data:
    - data_toy: Toy dataset samplers
    - data_image: Image dataset loaders

Training & Evaluation:
    - trainer: Training infrastructure
    - metrics: Evaluation metrics
    - visualization: Plotting utilities

Utilities:
    - utils: General utilities

Example Usage
-------------
>>> from ads_cft import (
...     ExperimentConfig,
...     create_model,
...     create_geometry,
...     Trainer,
... )
>>> 
>>> # Create configuration
>>> config = ExperimentConfig(
...     dataset="checkerboard",
...     slice_geometry="planar",
...     use_spectral_encoding=True,
... )
>>> 
>>> # Create and train model
>>> model = create_model(config)
>>> trainer = Trainer(config)
>>> trainer.train(model)

Adding New Components
---------------------
The registry system allows easy extension:

>>> from ads_cft.registry import register_geometry, register_laplacian
>>> 
>>> @register_geometry("my_geometry")
>>> class MyGeometry(AdSGeometry):
...     ...
>>> 
>>> @register_laplacian("my_laplacian")
>>> class MyLaplacian(SliceLaplacian):
...     ...

Then use via CLI: `python train.py --geometry my_geometry --laplacian my_laplacian`

Version
-------
__version__ = "1.0.0"

Authors
-------
Implementation for "Literal AdS/CFT-Sliced Generative Modelling via Bulk Flow Matching"

License
-------
MIT License
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "AdS/CFT Flow Matching Team"

# =============================================================================
# Core Configuration Classes
# =============================================================================

from ads_cft.config import (
    # Enums
    SliceGeometry,
    PathType,
    FieldEncodingType,
    DatasetType,
    ODESolver,
    # Geometry config
    GeometryConfig,
    # Discretization
    DiscretizationConfig,
    # Path config
    PathConfig,
    # Model config
    FlowModelConfig,
    # Training configs
    TrainerConfig,
    ExperimentConfig,
    BaselineConfig,
    # ODE config
    IntegratorConfig,
)

# =============================================================================
# Registry System
# =============================================================================

from ads_cft.registry import (
    # Decorators
    register_geometry,
    register_laplacian,
    register_encoder,
    register_dataset,
    register_solver,
    register_baseline,
    # Factories
    create_geometry,
    create_laplacian,
    create_encoder,
    create_dataset,
    create_solver,
    create_baseline,
    create_model,
    # Discovery
    list_available,
    print_registry_status,
)

# =============================================================================
# Geometry Classes
# =============================================================================

from ads_cft.geometry import (
    AdSGeometry,
    PlanarAdS,
    FlatGeometry,
    HyperscalingViolatingAdS,
    make_geometry,
)

# =============================================================================
# Public API Summary
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # Enums
    "SliceGeometry",
    "PathType",
    "FieldEncodingType",
    "DatasetType",
    "ODESolver",
    # Configs
    "GeometryConfig",
    "DiscretizationConfig",
    "PathConfig",
    "FlowModelConfig",
    "TrainerConfig",
    "ExperimentConfig",
    "BaselineConfig",
    "IntegratorConfig",
    # Registry decorators
    "register_geometry",
    "register_laplacian",
    "register_encoder",
    "register_dataset",
    "register_solver",
    "register_baseline",
    # Factories
    "create_geometry",
    "create_laplacian",
    "create_encoder",
    "create_dataset",
    "create_solver",
    "create_baseline",
    "create_model",
    # Discovery
    "list_available",
    "print_registry_status",
    # Geometry classes
    "AdSGeometry",
    "PlanarAdS",
    "FlatGeometry",
    "HyperscalingViolatingAdS",
    "make_geometry",
]