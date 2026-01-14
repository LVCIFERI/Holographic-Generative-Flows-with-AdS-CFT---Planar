# Holographic Emergence Generative Models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A physics-inspired deep generative model based on **Anti-de Sitter / Conformal Field Theory (AdS/CFT)** correspondence. This framework implements **bulk flow matching** in UV-stabilized variables, combining holographic principles with modern generative modeling techniques.

---

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Background](#mathematical-background)
   - [AdS/CFT Geometry](#adscft-geometry)
   - [Klein-Gordon Backbone](#klein-gordon-backbone)
   - [UV-Stabilized Variables](#uv-stabilized-variables)
   - [Hyperscaling-Violating Geometries](#hyperscaling-violating-geometries)
   - [Flow Matching Paths](#flow-matching-paths)
   - [Loss Function](#loss-function)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Project Structure](#project-structure)
6. [Command Reference](#command-reference)
7. [Configuration Reference](#configuration-reference)
8. [Datasets](#datasets)
9. [Geometries](#geometries)
10. [Encoding Types](#encoding-types)
11. [Laplacian Implementations](#laplacian-implementations)
12. [Advanced Usage](#advanced-usage)
13. [Evaluation](#evaluation)
14. [Citation](#citation)

---

## Overview

This framework implements a novel generative model that treats data generation as **holographic bulk evolution**. The key insight from AdS/CFT is that boundary data (our observations) can be reconstructed by evolving bulk fields from the IR (deep interior) to the UV (boundary).

### Key Features

- **Physics-Informed Architecture**: Klein-Gordon backbone respects AdS geometry
- **UV-Stabilized Variables**: Removes exponential growth/decay near boundary
- **Planar AdS Geometry**: Flat boundary with exponential warp factor
- **Hyperscaling-Violating (HSV)**: Interpolates between flat space and AdS
- **Flexible Paths**: Linear (robust) and Hermite (geometry-aware) interpolation
- **Multiple Laplacians**: Diagonal (FFT-based) implementation
- **Comprehensive Encodings**: Point, spectral, holographic, and image encodings

---

## Mathematical Background

This section details the AdS/CFT physics underlying the generative model. All formulas follow the conventions in the accompanying paper.

### AdS/CFT Geometry

The AdS metric in warped-product form is (paper eq. 1):

$$ds^2 = dr^2 + f(r)^2 \, \hat{g}_{ab} \, dx^a dx^b$$

where:
- $r$ is the **radial (holographic) coordinate**: flows from $r_{IR}$ (deep bulk) to $r_{UV}$ (near boundary)
- $f(r)$ is the **warp factor** determining the geometry
- $\hat{g}_{ab}$ is the **reference metric** on constant-$r$ slices (flat $\mathbb{R}^d$ for planar)

**Supported Geometries:**

| Geometry | Warp Factor $f(r)$ | $f'/f$ | Description |
|----------|-------------------|--------|-------------|
| **Planar** | $e^r$ | $1$ | Standard AdS with flat boundary $\mathbb{R}^d$ |
| **Flat** (ablation) | $1$ | $0$ | No AdS curvature (baseline) |
| **HSV** | $[(1-p)r]^{-p/(1-p)}$ | $-p/[(1-p)r]$ | Hyperscaling-violating, interpolates flat↔AdS |

The **slice measure** (volume element weight) is:
$$\omega(r) = f(r)^d$$

### Klein-Gordon Backbone

The bulk scalar field $\Phi$ satisfies the Klein-Gordon equation (paper eq. 3):

$$\left(\partial_r^2 + \frac{d f'(r)}{f(r)} \partial_r + \frac{1}{f(r)^2} \hat{\Delta}_g - m^2\right)\Phi = 0$$

In **first-order form** with canonical momentum $\Pi \equiv \partial_r \Phi$ (paper eqs. 16-17):

$$\frac{\partial \Phi}{\partial r} = \Pi$$

$$\frac{\partial \Pi}{\partial r} = \left(\Delta(\Delta-d) - \frac{1}{f^2} \hat{\Delta}_g\right)\Phi - \frac{d f'}{f} \Pi$$

where $\hat{\Delta}_g$ is the **Laplace-Beltrami operator** on the slice. For planar geometry with periodic boundary conditions, eigenvalues are $\lambda_k = |k|^2$ (computed via FFT).

The **mass-dimension relation** from AdS/CFT is:
$$m^2 = \Delta(\Delta - d)$$

with $\Delta$ being the **conformal dimension** of the dual CFT operator. Standard quantization requires $\Delta > d/2$.

### UV-Stabilized Variables

Near the UV boundary ($r \to \infty$), physical fields exhibit exponential behavior (paper eq. 4):
$$\Phi \sim e^{-(d-\Delta)r} J(x) + \frac{e^{-\Delta r}}{2\Delta - d} \langle O(x) \rangle$$

To handle this numerically, we introduce **UV-stabilized variables** (paper eq. 18):

$$\tilde{\Phi} = e^{\Delta r} \Phi$$

$$\tilde{\Pi} = e^{\Delta r} (\Pi + \Delta \Phi)$$

These satisfy the **stabilized backbone equations**:

$$\frac{\partial \tilde{\Phi}}{\partial r} = \tilde{\Pi}$$

$$\frac{\partial \tilde{\Pi}}{\partial r} = \left(2\Delta - d\frac{f'}{f}\right) \tilde{\Pi} - \frac{1}{f^2} \hat{\Delta}_g \tilde{\Phi} + d\Delta\left(\frac{f'}{f} - 1\right) \tilde{\Phi}$$

For **planar AdS** ($f'/f = 1$), this simplifies to (paper eqs. 20-21):

$$\frac{d\tilde{\phi}_k}{dr} = \tilde{\pi}_k$$

$$\frac{d\tilde{\pi}_k}{dr} = |k|^2 e^{-2r} \tilde{\phi}_k - (d - 2\Delta) \tilde{\pi}_k$$

### Bulk-Boundary Propagator

The bulk-boundary propagator $K(r,x;x')$ solves Klein-Gordon with delta-function boundary conditions. In Fourier space, the UV-stabilized mode coefficients are (paper eq. 22):

$$\tilde{\kappa}_{|k|}(r) = \frac{2}{\Gamma(\nu)} \left(\frac{|k| e^r}{2}\right)^\nu K_\nu(|k| e^{-r})$$

where $\nu = \Delta - d/2$ and $K_\nu$ is the modified Bessel function of the second kind.

### Hyperscaling-Violating Geometries

HSV geometries (paper Section 5) form a one-parameter family interpolating between flat space and AdS.

**Note on conventions:** The code uses $p \in [0, 1)$ where $p=0$ is flat and $p \to 1$ is AdS. This is **inverted** from the paper's convention.

**Warp factor:**
$$f(r) = [(1-p)r]^{-p/(1-p)}$$

**HSV coordinate:** The natural radial coordinate appearing in the propagator is:
$$u = [(1-p)r]^{1/(1-p)}$$

which satisfies $u = r$ when $p=0$ (flat) and approaches $u \to e^{-r} = z$ (Poincaré coordinate) as $p \to 1$.

**Bulk-boundary propagator:** In Fourier space, the HSV propagator has a unified Bessel form (paper eq. 47):

$$\hat{K}(u, k) = \frac{(|k| u)^\beta K_\beta(|k| u)}{2^{\beta-1} \Gamma(\beta)}$$

where the **Bessel order** is:
$$\beta = \frac{1 + (d-1)p}{2}$$

**Limiting cases:**
- $p = 0$: $\beta = 1/2$, $\hat{K} = e^{-|k|r}$ (flat space Green's function)
- $p \to 1$: $\beta \to d/2$, $\hat{K} \to (|k|z)^{d/2} K_{d/2}(|k|z)$ (standard AdS propagator)

### Flow Matching Paths

We connect IR prior states $S_0 \sim \mu_{IR}$ to UV data states $S_1 = \text{Lift}(y)$ via interpolation paths.

**Linear Path** (robust default):
$$S_t = (1-t) S_0 + t S_1, \quad U_t = S_1 - S_0$$

**AdS-Affine Hermite Path** (geometry-aware, paper Section 3.1):

Uses the affine parameter (paper eq. 29):
$$u(r) = \frac{1 - e^{-d(r - r_{IR})}}{1 - e^{-d(r_{UV} - r_{IR})}}$$

Cubic Hermite interpolation in $u$-space ensures $\tilde{\Pi}_t = \partial_r \tilde{\Phi}_t$ by construction, keeping the path **on-shell** (on the Lagrangian submanifold of phase space).

### Loss Function

The **backbone-subtracted flow matching loss** is (paper eq. 35):

$$\mathcal{L}(\theta) = \left\langle\left\| R_\theta(S_t, t) - (U_t - \delta_r V_{KG}) \right\|^2_{g_{r(t)}}\right\rangle_t$$

where:
- $V_{KG}$ is the Klein-Gordon backbone velocity
- $\delta_r = r_{UV} - r_{IR}$ is the radial extent
- $R_\theta = (0, N_\theta)$ for Hermite path (residual only affects $\tilde{\Pi}$)
- $R_\theta = (N_\theta^\phi, N_\theta^\pi)$ for linear path
- The norm uses the **intrinsic slice measure**: $\|\Psi\|^2_{g_r} = \int_\Sigma f(r)^d \left(|\Psi^{\tilde{\Phi}}|^2 + |\Psi^{\tilde{\Pi}}|^2\right) d\text{vol}_{\hat{g}}$

### Spectral Point Encoding

Data points $x_*$ are encoded as point sources in the CFT (paper Section 3.3):
$$J(x|x_*) = \mathcal{N}_\nu \delta(x - x_*)$$

This creates bulk field configurations via the propagator:
$$\Phi(r,x|x_*) = \mathcal{N}_\nu K(r,x;x_*)$$

In Fourier space, the phase encodes position:
$$\tilde{\phi}_k(r|x_*) = \frac{\mathcal{N}_\nu}{(2\pi)^{d/2}} \tilde{\kappa}_{|k|}(r) e^{ik \cdot x_*}$$

### Algorithms

**Algorithm 1: Training**
```
1. Sample y ~ p_data, S_0 ~ μ_IR
2. Set S_1 = Lift(y)
3. Sample t ~ Uniform[0, 1]
4. Compute S_t, U_t via path interpolation
5. Compute backbone V_KG(S_t, r_t)
6. Train: minimize ||R_θ(S_t, t) - (U_t - V_KG)||²_{g_{r(t)}}
```

**Algorithm 2: Sampling**
```
1. Sample S(r_IR) ~ μ_IR
2. Integrate dS/dr = V_KG(S, r) + R_θ(S, r) from r_IR to r_UV
3. Return y_gen = Readout(S(r_UV))
```

---

## Installation

### Requirements

```bash
pip install torch>=2.0.0 torchvision numpy scipy matplotlib tqdm
```

### Optional Dependencies

```bash
# For advanced metrics
pip install pot  # Python Optimal Transport
```

### Install the Framework

**Option 1: Development Install (Recommended)**

```bash
git clone https://github.com/VargEM/Holographic-Emergence-Generative-Models.git
cd Holographic-Emergence-Generative-Models
pip install -e .
```

This installs the `ads_cft` package in development mode, allowing you to edit the code while using it.

**Option 2: Add to PYTHONPATH**

```bash
export PYTHONPATH=/path/to/Holographic-Emergence-Generative-Models:$PYTHONPATH
```

After installation, you can import the package:
```python
import ads_cft
from ads_cft.model import UVStabilizedFlowMatchingModel
```

---

## Quick Start

### Basic Training

```bash
# Train on checkerboard with default settings (Hermite path, planar geometry)
python -m ads_cft.train --dataset checkerboard --epochs 100

# Train with spectral encoding (recommended)
python -m ads_cft.train --dataset checkerboard --use_spectral_encoding --epochs 100

# Train on Gaussian mixture with linear path
python -m ads_cft.train --dataset gaussian_mixture --path_type linear --epochs 100

# Train with HSV geometry (p=0.5 interpolates between flat and AdS)
python -m ads_cft.train --dataset checkerboard --slice_geometry planar_hsv --hsv_p 0.5 --epochs 100

# Train on MNIST
python -m ads_cft.train --dataset mnist --use_image_encoding --epochs 50
```

---

## Project Structure

```
ads_cft/
├── __init__.py              # Package exports
├── config.py                # All configuration dataclasses
├── registry.py              # Component registration system
│
├── geometry.py              # AdS geometries (PlanarAdS, FlatGeometry, HyperscalingViolatingAdS)
│
├── laplacian_base.py        # Slice Laplacian (PlanarSpectralLaplacian, SpectralLaplacian)
│
├── encoding_base.py         # Encoder protocols and Bessel function helpers
├── encoding_spectral.py     # Spectral holographic encoders (SpectralHolographicEncoder)
├── encoding_image.py        # Image spectral encoders
│
├── networks.py              # Neural network components (embeddings, backbones)
├── model.py                 # Main UVStabilizedFlowMatchingModel
├── baselines.py             # Baseline comparison models (MLP, UNet)
│
├── data_toy.py              # Toy dataset samplers (checkerboard, GMM, swiss_roll, etc.)
├── data_image.py            # Image dataset loaders (MNIST, CIFAR)
│
├── train.py                 # Training script
├── evaluate.py              # Evaluation and metrics
├── visualization.py         # Plotting utilities
├── utils.py                 # Helper functions
│
├── run_experiments.sh       # Experiment runner script
├── run_hsv_experiments.sh   # HSV geometry experiments
└── test_planar_geometry_cleanup.py  # Validation tests
```

---

## Command Reference

### train.py

```bash
python train.py [options]
```

**Core Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset` | `checkerboard` | Dataset name |
| `--slice_geometry` | `planar` | Geometry: `planar`, `flat`, `planar_hsv` |
| `--path_type` | `hermite` | Path interpolation: `hermite`, `linear` |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `64` | Batch size |
| `--lr` | `3e-4` | Learning rate |

**Geometry Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--d` | `2` | Boundary spatial dimension |
| `--r_ir` | `0.0` | IR radial coordinate |
| `--r_uv` | `1.0` | UV radial coordinate |
| `--deltas` | `1.5 1.5` | Conformal dimensions (must be > d/2) |

**HSV Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--hsv_p` | `0.5` | HSV parameter p ∈ [0, 1): p=0 flat, p→1 AdS |
| `--hsv_use_u_bounds` | `True` | Use natural HSV u-coordinates for bounds |
| `--hsv_u_uv` | `0.1` | HSV u at UV boundary |
| `--hsv_u_ir` | `1.0` | HSV u at IR boundary |
| `--hsv_ads_threshold` | `0.95` | p threshold for switching to AdS formulas |

**Spectral Encoding Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--use_spectral_encoding` | `False` | Enable spectral bulk-boundary encoding |
| `--spectral_n_modes` | `16` | Number of Fourier modes per dimension |
| `--spectral_domain` | `planar` | Spectral domain type |

**Laplacian Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--laplacian_type` | `diagonal` | Laplacian: `diagonal` (FFT-based) |

**ODE Solver Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--ode_solver` | `rk4` | Solver: `euler`, `heun`, `midpoint`, `rk4`, `leapfrog`, `implicit_midpoint` |
| `--ode_n_steps` | `50` | Number of integration steps |
| `--residual_type` | `direct` | Residual type: `direct`, `potential` |

**Network Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--residual_hidden` | `64` | Hidden dimension |
| `--residual_depth` | `3` | Network depth |
| `--residual_time_emb_dim` | `128` | Time embedding dimension |

---

## Configuration Reference

### Geometry Configurations

**Planar AdS** (default):
```python
GeometryConfig(
    slice_geometry=SliceGeometry.PLANAR,
    d=2,
    r_ir=0.0,
    r_uv=1.0,
)
```

**Flat** (ablation baseline):
```python
GeometryConfig(
    slice_geometry=SliceGeometry.FLAT,
    d=2,
    r_ir=0.0,
    r_uv=1.0,
)
```

**HSV** (hyperscaling-violating):
```python
GeometryConfig(
    slice_geometry=SliceGeometry.HYPERSCALING_VIOLATING,
    d=2,
    hsv_p=0.5,  # 0=flat, 1=AdS
    hsv_use_u_bounds=True,
    hsv_u_uv=0.1,
    hsv_u_ir=1.0,
)
```

---

## Datasets

### Toy Datasets

| Dataset | Description |
|---------|-------------|
| `checkerboard` | 2D checkerboard pattern |
| `gaussian_mixture` | Mixture of Gaussians |
| `swiss_roll` | Swiss roll manifold |
| `two_moons` | Two interleaving half-circles |
| `concentric_circles` | Concentric ring patterns |
| `pinwheel` | Pinwheel/spiral clusters |

### Image Datasets

| Dataset | Description |
|---------|-------------|
| `mnist` | MNIST handwritten digits (28×28) |
| `cifar10` | CIFAR-10 images (32×32) |

---

## Geometries

### Planar AdS

Standard AdS geometry with flat $\mathbb{R}^d$ boundary slicing.

**Warp factor:** $f(r) = e^r$

**Properties:**
- $f'/f = 1$ (constant)
- Slice Laplacian: $\Delta_{\hat{g}} = \sum_i \partial_i^2$ (flat Laplacian)
- Eigenvalues: $\lambda_k = |k|^2$ (via FFT)

**Usage:**
```bash
python train.py --slice_geometry planar
```

### Flat (Ablation)

No AdS curvature - serves as ablation baseline.

**Warp factor:** $f(r) = 1$

**Properties:**
- $f'/f = 0$
- No holographic warping
- Tests whether AdS structure improves generation

**Usage:**
```bash
python train.py --slice_geometry flat
```

### Hyperscaling-Violating (HSV)

Interpolates between flat space and AdS via parameter $p \in [0, 1)$.

**Warp factor:** $f(r) = [(1-p)r]^{-p/(1-p)}$

**Properties:**
- $p = 0$: flat space ($f = 1$)
- $p \to 1$: recovers AdS ($f \to e^r$)
- Bessel-based propagator with order $\beta = (1 + (d-1)p)/2$

**Usage:**
```bash
# Halfway between flat and AdS
python train.py --slice_geometry planar_hsv --hsv_p 0.5

# Nearly AdS
python train.py --slice_geometry planar_hsv --hsv_p 0.9

# Sweep p values
./run_hsv_experiments.sh checkerboard 100
```

---

## Encoding Types

### Point Encoding (Default)

Direct representation of point data without field structure.

**Usage:**
```bash
python train.py --dataset checkerboard  # No encoding flags
```

### Spectral Encoding (Recommended)

Encodes boundary data into spectral (Fourier) bulk fields using the holographic bulk-boundary propagator.

**Mathematical basis:**
$$\hat{\Phi}(r, k) = \hat{K}_\Delta(r, k) \cdot \hat{\phi}_{boundary}(k)$$

where $\hat{K}_\Delta$ is the UV-stabilized propagator:
$$\hat{K}_\Delta(r, k) = \frac{2}{\Gamma(\nu)} e^{\nu r} \left(\frac{|k|}{2}\right)^\nu K_\nu(|k| e^{-r})$$

with $\nu = \Delta - d/2$.

**Usage:**
```bash
python train.py --use_spectral_encoding --spectral_n_modes 16
```

### Image Encoding

Specialized encoding for image datasets, treating images as boundary field configurations.

**Usage:**
```bash
python train.py --dataset mnist --use_image_encoding
```

---

## Laplacian Implementations

### Diagonal (Default)

FFT-based diagonal Laplacian for planar geometry. Exact and efficient.

**Complexity:** $O(N \log N)$ via FFT

**Usage:**
```bash
python train.py --laplacian_type diagonal
```

---

## Advanced Usage

### HSV Geometry Sweep

```bash
# Run comprehensive HSV experiments
./run_hsv_experiments.sh checkerboard 100

# Specific p values
python train.py --slice_geometry planar_hsv --hsv_p 0.0   # Flat
python train.py --slice_geometry planar_hsv --hsv_p 0.25  # Quarter
python train.py --slice_geometry planar_hsv --hsv_p 0.5   # Half
python train.py --slice_geometry planar_hsv --hsv_p 0.75  # Three-quarters
python train.py --slice_geometry planar_hsv --hsv_p 0.9   # Near-AdS
```

### Symplectic Integration

For energy-conserving dynamics:

```bash
python train.py \
    --ode_solver leapfrog \
    --residual_type potential \
    --use_spectral_encoding
```

### Conformal Dimension Sweep

```bash
# Test different conformal dimensions (must be > d/2 = 1.0 for d=2)
python train.py --deltas 1.5 1.5  # Default
python train.py --deltas 2.0 2.0  # Higher
python train.py --deltas 2.5 2.5  # Even higher
```

### Full Training Pipeline

```bash
# 1. Train
python train.py \
    --name gmm_spectral \
    --dataset gaussian_mixture \
    --use_spectral_encoding \
    --spectral_n_modes 32 \
    --path_type hermite \
    --epochs 500 \
    --batch_size 128

# 2. Evaluate
python evaluate.py --experiment gmm_spectral --n_samples 10000

# 3. Generate samples
python -c "
import torch
from ads_cft.model import UVStabilizedFlowMatchingModel

checkpoint = torch.load('results/gmm_spectral/best_model.pt')
model = UVStabilizedFlowMatchingModel(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

samples = model.sample(1000)
print(f'Generated {samples.shape[0]} samples')
"
```

---

## Evaluation

### evaluate.py

```bash
python evaluate.py [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | `all` | `all`, `toy`, or `image` |
| `--results_dir` | `results` | Results directory |
| `--experiment` | None | Specific experiment to evaluate |
| `--n_samples` | `10000` | Number of samples to generate |
| `--device` | `cuda` | Device to use |

**Examples:**
```bash
# Evaluate all experiments
python evaluate.py --mode all

# Evaluate specific experiment
python evaluate.py --experiment ads_checkerboard_hermite_planar

# Evaluate with more samples
python evaluate.py --n_samples 50000
```

---

## Ablation Studies

```bash
./run_experiments.sh ablation
```

### What Ablation Mode Tests

| Category | Experiments | Purpose |
|----------|-------------|---------|
| **ODE Solvers** | `rk4`, `euler`, `heun`, `midpoint` | Compare integrator accuracy |
| **Symplectic Solvers** | `leapfrog`, `implicit_midpoint` | Energy conservation |
| **Delta Values** | `1.5`, `2.0`, `2.5`, `3.0` | Conformal dimension sensitivity |

---

## Citation

If you use this code, please cite:

```bibtex
@article{adscft_flow_matching,
  title={UV-Stabilized Generative Flow Matching via AdS/CFT},
  author={...},
  journal={...},
  year={2024}
}
```

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

This implementation is based on the mathematical framework of AdS/CFT correspondence applied to generative modeling, combining insights from theoretical physics with modern deep learning techniques.
