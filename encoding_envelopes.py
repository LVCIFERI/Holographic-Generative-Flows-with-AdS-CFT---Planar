"""
encoding_envelopes.py

Generic spectral envelope profiles for the referee-requested control experiments
================================================================================

Referee request (Report 1, experimental attribution):

  (a) "a spectral CNN baseline that uses the same Fourier representation of the
      data, the same phase-space dimension, the same network architecture,
      optimizer, and training budget, but WITHOUT the AdS propagator envelope,
      without the warped loss weighting, and without the Klein-Gordon backbone."

  (b) "compare the AdS spectral envelope with at least one generic coarse-to-fine
      spectral filter, for example a heat-kernel/Gaussian filter or a matched
      Matern-type filter, implemented in the same pipeline."

This module implements the envelope profiles for (a) and (b) so that the ONLY
thing that changes relative to the published "AdS" model is the two envelope
tensors used by ``SpectralHolographicEncoder``.  Everything else (Fourier phase
encoding, phase-space layout, CNN, optimizer, decode, IR prior, ODE sampler)
is untouched.

Design
------
All profiles share the SAME radial schedule as the AdS propagator,

    xi(r, k) = |k| * exp(-r),

so the coarse-to-fine *schedule* is identical across models and only the filter
*shape* differs.  This is the controlled comparison the referee asks for: it
isolates the AdS-specific Bessel shape from generic spectral smoothing.

Profiles (phi envelope g(xi), pi envelope = d/dr g(xi(r,k)) evaluated at r_UV):

  "ads"    (reference, identical to encoding_spectral._init_planar):
       g(xi)      = xi^nu K_nu(xi) / N ,        nu = Delta - d/2
       pi(xi)     = xi^{nu+1} K_{nu-1}(xi) / N
     using  d/dr [xi^nu K_nu(xi)] = + xi^{nu+1} K_{nu-1}(xi)   (d xi/dr = -xi).
     N is the value of the raw phi profile at the k=0 grid mode, exactly as in
     the published code, so g -> 1 as xi -> 0.

  "heat"   (heat-kernel / Gaussian low-pass, referee's first suggestion):
       g(xi; a)   = exp(-a xi^2)
       d/dr g     = (-2 a xi) * (d xi / dr) * g = + 2 a xi^2 exp(-a xi^2)
     The single free parameter a > 0 is matched to the AdS envelope (below).

  "matern" (matched Matern-type filter, referee's second suggestion):
     The Matern spectral density with smoothness nu (the SAME nu = Delta - d/2
     as the AdS propagator, i.e. matched smoothness) and length scale l is
       S(k) ~ (2 nu / l^2 + |k|^2)^{-(nu + d/2)} .
     Normalised to 1 at k = 0 and expressed in the shared variable xi:
       g(xi; l)   = (1 + (l xi)^2 / (2 nu))^{-s},    s = nu + d/2
       d/dr g     = + s * (l^2 xi^2 / nu) * (1 + (l xi)^2/(2 nu))^{-(s+1)}
     The single free parameter l > 0 is matched to the AdS envelope (below).

  "none"   (no envelope; for the physics-free spectral CNN baseline):
       g(xi)      = 1     for every mode  (identity: plain Fourier phases)
       pi profile = 0     so that, after the standard momentum lift noise,
                          Pi-tilde is purely ancillary noise of scale
                          ``lift_noise_sigma`` -- the same convention used for
                          the published FCN baseline (points) and the vanilla
                          CNN baseline (MNIST).  The phase-space dimension and
                          the network input/output are therefore IDENTICAL to
                          the AdS models, as the referee requires.

Parameter matching ("matched ... filter")
-----------------------------------------
Two deterministic matching modes are provided:

  match = "lsq"   (default; recommended, used by run_referee_experiments.sh)
     The free parameter (a or l) is fit by least squares against the AdS phi
     envelope over the ACTUAL K x K mode grid used in the experiment:
         theta* = argmin_theta  sum_j [ g(xi_j; theta) - g_AdS(xi_j) ]^2 .
     Each Fourier mode enters exactly once, i.e. with the same multiplicity
     with which it enters the training loss.  The fit is a 1-D deterministic
     optimisation (coarse log-space scan + bounded refinement); the fitted
     parameter and the residual RMS are reported so that the comparison is
     fully auditable ("the generic filter is as close to the AdS envelope as
     a one-parameter family allows; any performance difference is then
     attributable to the residual shape difference").

  match = "efold"
     The free parameter is chosen so that the generic envelope crosses 1/e at
     the same wavenumber xi* at which the AdS envelope crosses 1/e:
         g_AdS(xi*) = 1/e  and  g(xi*; theta) = 1/e .
     (Closed form:  heat  a = 1/xi*^2 ;
                    matern l = sqrt(2 nu (e^{1/s} - 1)) / xi* .)

Numerical notes
---------------
* For Delta = 1.5, d = 2 (the setting of all published checkerboard runs) the
  AdS phi envelope is EXACTLY exp(-xi): xi^{1/2} K_{1/2}(xi) = sqrt(pi/2) e^{-xi}.
  This is used as an internal self-check (see verify_ads_closed_form).
* The k = 0 grid mode uses the same epsilon-regularised |k| as the published
  encoder (k_mag = sqrt(k^2 + regularization^2)), and every profile is divided
  by its own value at that mode, mirroring the published normalisation.
* Everything is computed in float64 and cast to the requested dtype at the end.

This module has no state; it only produces the two envelope tensors consumed by
``SpectralHolographicEncoder``.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

Tensor = torch.Tensor

VALID_ENVELOPE_TYPES = ("ads", "heat", "matern", "none")
VALID_MATCH_MODES = ("lsq", "efold")


# =============================================================================
# AdS reference profile (identical maths to encoding_spectral._init_planar)
# =============================================================================


def _ads_phi_pi_profiles(
    xi: Tensor,
    nu: float,
    regularization: float,
) -> Tuple[Tensor, Tensor]:
    """
    Raw (unnormalised) AdS UV-stabilised propagator profiles on a xi grid.

        phi_raw(xi) = xi^nu     K_nu(xi)
        pi_raw(xi)  = xi^{nu+1} K_{nu-1}(xi)      [= d/dr phi_raw, since dxi/dr = -xi]

    Uses the SAME Bessel implementation as the published encoders so that the
    "ads" reference used for matching is bit-compatible with the model that the
    generic filters are compared against.
    """
    from ads_cft.encoding_base import _bessel_k_nu

    K_nu = _bessel_k_nu(nu, xi, eps=regularization)
    K_nu_m1 = _bessel_k_nu(abs(nu - 1.0), xi, eps=regularization)
    phi_raw = torch.pow(xi, nu) * K_nu
    pi_raw = torch.pow(xi, nu) * xi * K_nu_m1
    return phi_raw, pi_raw


def ads_reference_envelopes(
    k_mag: Tensor,
    delta: float,
    d: int,
    r_uv: float,
    regularization: float,
) -> Tuple[Tensor, Tensor]:
    """
    Normalised AdS (phi, pi) envelopes on the experiment's k grid.

    Identical, term by term, to SpectralHolographicEncoder._init_planar:
    both profiles are divided by the raw phi profile at the k=0 grid mode.

    Args:
        k_mag: (K, K) regularised |k| grid (as stored on the encoder).
        delta: conformal dimension Delta of the channel.
        d: boundary dimension.
        r_uv: UV radius at which the lift is evaluated.
        regularization: epsilon used inside the Bessel evaluation.

    Returns:
        (phi_env, pi_env), same shape/dtype as k_mag.
    """
    nu = float(delta) - d / 2.0
    xi = (k_mag.to(torch.float64) * math.exp(-r_uv)).clamp_min(1e-300)
    phi_raw, pi_raw = _ads_phi_pi_profiles(xi, nu, regularization)
    norm = phi_raw.reshape(-1)[0]  # k = 0 grid mode, as in the published code
    if torch.abs(norm) > 1e-10:
        phi_env = phi_raw / norm
        pi_env = pi_raw / norm
    else:  # pragma: no cover - matches published fallback
        phi_env, pi_env = phi_raw, pi_raw
    return phi_env.to(k_mag.dtype), pi_env.to(k_mag.dtype)


def verify_ads_closed_form(
    k_mag: Tensor, r_uv: float, regularization: float, atol: float = 1e-4
) -> float:
    """
    Self-check: for Delta = 1.5, d = 2 the AdS phi envelope is exp(-xi) up to
    normalisation, i.e. env(xi_i)/env(xi_j) = exp(-(xi_i - xi_j)).

    The published Bessel helper clamps its argument at eps=regularization, so
    the check is performed on the unclamped region xi > 2*eps using ratios
    (which cancel the shared normalisation at the clamped k=0 grid mode).

    Returns the maximum absolute ratio deviation (raises if above atol).
    Called by the test-suite / smoke tests; not used in training.
    """
    phi_env, _ = ads_reference_envelopes(k_mag, 1.5, 2, r_uv, regularization)
    xi = (k_mag.to(torch.float64) * math.exp(-r_uv)).reshape(-1)
    env = phi_env.to(torch.float64).reshape(-1)
    mask = xi > 2.0 * regularization
    if int(mask.sum()) < 2:
        raise AssertionError("Too few unclamped modes for the closed-form check")
    xi_u, env_u = xi[mask], env[mask]
    ratio = env_u / env_u[0]
    exact = torch.exp(-(xi_u - xi_u[0]))
    dev = float((ratio - exact).abs().max())
    if dev > atol:
        raise AssertionError(
            f"AdS closed-form self-check failed: max deviation {dev:.3e} > {atol}"
        )
    return dev


# =============================================================================
# Generic profiles g(xi; theta) and their radial derivatives
# =============================================================================


def _heat_profiles(xi: Tensor, a: float) -> Tuple[Tensor, Tensor]:
    """Heat/Gaussian filter: g = exp(-a xi^2), d/dr g = +2 a xi^2 g."""
    g = torch.exp(-a * xi * xi)
    dg_dr = 2.0 * a * xi * xi * g
    return g, dg_dr


def _matern_profiles(xi: Tensor, ell: float, nu: float, d: int) -> Tuple[Tensor, Tensor]:
    """
    Matched Matern-type filter with smoothness nu = Delta - d/2 (same as AdS).

        s = nu + d/2
        g       = (1 + (ell xi)^2 / (2 nu))^{-s}
        d/dr g  = + s * (ell^2 xi^2 / nu) * (1 + (ell xi)^2 / (2 nu))^{-(s+1)}
    """
    if nu <= 0:
        raise ValueError(
            f"Matern-type filter requires nu = Delta - d/2 > 0, got nu = {nu}"
        )
    s = nu + d / 2.0
    u = (ell * xi) ** 2 / (2.0 * nu)
    base = 1.0 + u
    g = base ** (-s)
    dg_dr = s * (ell * ell * xi * xi / nu) * base ** (-(s + 1.0))
    return g, dg_dr


# =============================================================================
# Parameter matching
# =============================================================================


def _lsq_fit_log_parameter(
    profile_fn,
    xi_flat: Tensor,
    target_flat: Tensor,
    log_lo: float = -8.0,
    log_hi: float = 8.0,
    n_scan: int = 400,
) -> Tuple[float, float]:
    """
    Deterministic 1-D least-squares fit of a positive parameter.

    Minimises  F(theta) = sum_j [ g(xi_j; theta) - target_j ]^2  over
    theta = exp(t), t in [log_lo, log_hi]:  dense log-space scan followed by a
    bounded scalar refinement around the best scan point.

    Returns:
        (theta_star, rms_residual)
    """
    ts = torch.linspace(log_lo, log_hi, n_scan, dtype=torch.float64)

    def objective(t: float) -> float:
        g, _ = profile_fn(xi_flat, math.exp(t))
        return float(((g - target_flat) ** 2).sum())

    vals = torch.tensor([objective(float(t)) for t in ts], dtype=torch.float64)
    i = int(torch.argmin(vals))
    lo = float(ts[max(i - 1, 0)])
    hi = float(ts[min(i + 1, n_scan - 1)])

    try:
        from scipy.optimize import minimize_scalar

        res = minimize_scalar(
            objective, bounds=(lo, hi), method="bounded",
            options={"xatol": 1e-10, "maxiter": 200},
        )
        t_star = float(res.x)
    except Exception:  # pragma: no cover - scipy is a hard dependency in practice
        # Golden-section fallback (deterministic)
        phi = (math.sqrt(5.0) - 1.0) / 2.0
        a, b = lo, hi
        c, dpt = b - phi * (b - a), a + phi * (b - a)
        fc, fd = objective(c), objective(dpt)
        for _ in range(200):
            if fc < fd:
                b, dpt, fd = dpt, c, fc
                c = b - phi * (b - a)
                fc = objective(c)
            else:
                a, c, fc = c, dpt, fd
                dpt = a + phi * (b - a)
                fd = objective(dpt)
            if abs(b - a) < 1e-10:
                break
        t_star = 0.5 * (a + b)

    theta_star = math.exp(t_star)
    g_star, _ = profile_fn(xi_flat, theta_star)
    rms = float(torch.sqrt(((g_star - target_flat) ** 2).mean()))
    return theta_star, rms


def _ads_efold_crossing(
    nu: float, xi0: float, xi_max: float, regularization: float
) -> float:
    """
    Find xi* with  env_AdS(xi*) = 1/e, where env_AdS is normalised EXACTLY as
    the model normalises it: divided by the raw profile at the k = 0 grid mode
    xi0 (whose Bessel argument is clamped at eps = regularization, as in the
    published `_bessel_k_nu`).

    Dense 1-D grid + linear interpolation of the crossing; fully deterministic.
    """
    xi = torch.linspace(
        float(xi0), max(2.0 * xi_max, 10.0), 200_000, dtype=torch.float64
    )
    phi_raw, _ = _ads_phi_pi_profiles(xi, nu, regularization)
    g = phi_raw / phi_raw[0]  # phi_raw[0] is the value at xi0 == grid k=0 mode
    target = math.exp(-1.0)
    below = torch.nonzero(g <= target, as_tuple=False)
    if below.numel() == 0:
        raise RuntimeError("AdS envelope never crosses 1/e on the search grid.")
    j = int(below[0])
    if j == 0:
        return float(xi[0])
    # linear interpolation between (xi[j-1], g[j-1]) and (xi[j], g[j])
    x0, x1 = float(xi[j - 1]), float(xi[j])
    y0, y1 = float(g[j - 1]), float(g[j])
    w = (y0 - target) / max(y0 - y1, 1e-300)
    return x0 + w * (x1 - x0)


# =============================================================================
# Public API
# =============================================================================


def build_generic_envelopes(
    k_mag: Tensor,
    delta: float,
    d: int,
    r_uv: float,
    regularization: float,
    envelope_type: str,
    match: str = "lsq",
) -> Tuple[Tensor, Tensor, Dict[str, float]]:
    """
    Build (phi, pi) envelopes on the experiment's k grid for a generic profile.

    Args:
        k_mag: (K, K) regularised |k| grid (encoder buffer ``k_mag``).
        delta: conformal dimension of the channel (sets nu for "ads"/"matern"
            and the AdS reference used for matching).
        d: boundary dimension.
        r_uv: UV radius of the lift.
        regularization: epsilon used in the Bessel evaluation / k regularisation.
        envelope_type: one of {"ads", "heat", "matern", "none"}.
        match: "lsq" (least squares on the actual grid; default) or "efold".

    Returns:
        (phi_env, pi_env, info) with tensors shaped like ``k_mag`` and an info
        dict recording the matching diagnostics (fitted parameter, residual
        RMS against the AdS envelope, e-fold crossing, ...).
    """
    if envelope_type not in VALID_ENVELOPE_TYPES:
        raise ValueError(
            f"envelope_type must be one of {VALID_ENVELOPE_TYPES}, got {envelope_type!r}"
        )
    if match not in VALID_MATCH_MODES:
        raise ValueError(f"match must be one of {VALID_MATCH_MODES}, got {match!r}")

    dtype_out = k_mag.dtype
    xi = (k_mag.to(torch.float64) * math.exp(-r_uv)).clamp_min(1e-300)
    xi0 = xi.reshape(-1)[0]  # regularised k = 0 grid mode
    nu = float(delta) - d / 2.0

    info: Dict[str, float] = {
        "envelope_type_id": float(VALID_ENVELOPE_TYPES.index(envelope_type)),
        "delta": float(delta),
        "nu": nu,
        "r_uv": float(r_uv),
        "xi_min": float(xi.min()),
        "xi_max": float(xi.max()),
    }

    if envelope_type == "ads":
        phi_env, pi_env = ads_reference_envelopes(
            k_mag, delta, d, r_uv, regularization
        )
        return phi_env, pi_env, info

    if envelope_type == "none":
        # Identity envelope: plain Fourier phases, momentum purely ancillary.
        phi_env = torch.ones_like(k_mag)
        pi_env = torch.zeros_like(k_mag)
        info["note"] = 0.0  # placeholder to keep dict numeric-friendly
        return phi_env, pi_env, info

    # ---- matched generic filters: need the AdS reference on this grid --------
    ads_phi, _ = ads_reference_envelopes(k_mag, delta, d, r_uv, regularization)
    ads_phi64 = ads_phi.to(torch.float64).reshape(-1)
    xi_flat = xi.reshape(-1)

    if envelope_type == "heat":
        profile_fn = lambda x, a: _heat_profiles(x, a)  # noqa: E731
    else:  # "matern"
        profile_fn = lambda x, l: _matern_profiles(x, l, nu, d)  # noqa: E731

    if match == "lsq":
        theta, rms = _lsq_fit_log_parameter(profile_fn, xi_flat, ads_phi64)
        info["matched_parameter"] = theta
        info["lsq_rms_vs_ads"] = rms
    else:  # "efold"
        xi_star = _ads_efold_crossing(
            nu, float(xi0), float(xi.max()), regularization
        )
        info["xi_efold"] = xi_star
        if envelope_type == "heat":
            theta = 1.0 / (xi_star * xi_star)
        else:
            s = nu + d / 2.0
            theta = math.sqrt(2.0 * nu * (math.exp(1.0 / s) - 1.0)) / xi_star
        g_chk, _ = profile_fn(xi_flat, theta)
        info["matched_parameter"] = theta
        info["lsq_rms_vs_ads"] = float(
            torch.sqrt(((g_chk - ads_phi64) ** 2).mean())
        )

    g, dg_dr = profile_fn(xi, theta)

    # Normalise by the value at the k = 0 grid mode, mirroring the published
    # AdS normalisation (numerically ~1 here since xi0 is epsilon-small).
    g0, _ = profile_fn(xi0.reshape(1), theta)
    norm = g0.reshape(())
    if torch.abs(norm) > 1e-10:
        g = g / norm
        dg_dr = dg_dr / norm

    return g.to(dtype_out), dg_dr.to(dtype_out), info


def format_envelope_info(envelope_type: str, match: str, info: Dict[str, float]) -> str:
    """One-line human-readable summary for logs (captured in train.log)."""
    if envelope_type == "ads":
        return "[ENVELOPE] type=ads (published Bessel propagator envelope)"
    if envelope_type == "none":
        return (
            "[ENVELOPE] type=none (identity phi envelope, zero pi profile: "
            "Pi-tilde is ancillary lift noise; physics-free spectral baseline)"
        )
    p = info.get("matched_parameter", float("nan"))
    rms = info.get("lsq_rms_vs_ads", float("nan"))
    name = "a" if envelope_type == "heat" else "ell"
    return (
        f"[ENVELOPE] type={envelope_type} match={match} "
        f"{name}={p:.6g} (Delta={info.get('delta')}, nu={info.get('nu')}), "
        f"RMS vs AdS envelope over grid = {rms:.4g}"
    )
