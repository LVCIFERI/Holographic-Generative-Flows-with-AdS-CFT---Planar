# Referee-Requested Experiments — Implementation Guide

This package implements every **new experiment** requested in the two referee
reports for *Holographic generative flows with AdS/CFT* (MLST-105546), on top
of the existing framework, with no changes to any published experiment's
behaviour (all new options default to the published configuration).

---

## 1. Files

### New files (add to the repository root, next to `train.py`)

| File | Purpose | Referee request |
|---|---|---|
| `encoding_envelopes.py` | Generic spectral envelope profiles (`heat`, `matern`, `none`) matched to the AdS propagator envelope; full math + deterministic least-squares / e-fold matching. | R1 §experiments (a): spectral CNN baseline without the AdS envelope; (b): AdS envelope vs generic coarse-to-fine filters. R2: "what does the AdS propagator buy beyond a generic multiscale spectral parameterization?" |
| `metrics_mnist.py` | MNIST-classifier-feature FID, KID (classifier **and** Inception feature spaces, unbiased block estimator), improved precision/recall (Kynkäänniemi et al. 2019). | R1: "supplement FID with an MNIST-appropriate classifier-feature FID, KID, precision/recall-style coverage metrics". |
| `evaluate_mnist_extended.py` | Driver: computes the above per model per seed from saved samples **or** checkpoints (no retraining), aggregates mean ± std, prints an aligned table + LaTeX rows. | Same as above. |
| `run_referee_experiments.sh` | Checkerboard: [1] spectral CNN baseline, [2] heat-filter model, [3] Matérn-filter model, [4] lift-noise grid σ ∈ {0, 0.05, 0.2} × {hermite, linear, nokg} — all 3 seeds, full provenance. | R1 baseline + filter comparison + "quantify how results change when the lift noise is varied or set to zero". |
| `run_delta_sweep_multiseed.sh` | Δ ∈ {1.5, 2.0, 2.5, 3.0} × 3 seeds, original hyperparameters verbatim, per-Δ mean ± std. | R1: "Δ and HSV scans should be reported over multiple seeds with uncertainties". |
| `run_hsv_multiseed.sh` | p ∈ {0.1, 0.25, 0.5, 1.0} × 3 seeds, original hyperparameters verbatim, per-p mean ± std. | Same. |
| `run_mnist_noise_ablation.sh` | Optional MNIST arm of the lift-noise ablation (σ = 0, hermite + linear, 3 seeds, full 1500-epoch budget). | R1 lift-noise request (MNIST spot-check). |

### Modified files (drop-in replacements at the repository root)

| File | Change |
|---|---|
| `encoding_spectral.py` | `SpectralHolographicEncoder`/`Codec` accept `envelope_type` ∈ {ads, heat, matern, none} and `envelope_match` ∈ {lsq, efold}; for non-"ads" (planar only) the two envelope buffers are overwritten by `encoding_envelopes.py` — grid, phases, Laplacian, decode, prior and everything downstream are untouched. Default `"ads"` reproduces the published model bit-for-bit (verified). |
| `config.py` | Two new fields (`spectral_envelope_type`, `spectral_envelope_match`) in both `FlowModelConfig` and `ExperimentConfig`, defaulting to the published behaviour. |
| `model.py` | Passes the two fields into the spectral codec (4 lines). |
| `train.py` | (i) two new CLI flags; (ii) fields forwarded into `FlowModelConfig`; (iii) writes `git_commit.txt` next to the already-saved `config.json` (R1: immutable commit + complete configurations); (iv) the final EMA evaluation now saves the exact generated sample tensor to `samples/samples_final.pt`, so any present or future metric can be recomputed without retraining. |

Nothing else in the repository is touched. With the new flags left at their
defaults, every code path is identical to the published one (the smoke tests
below include a bit-equality check of the default envelopes).

## 2. Installation

Copy all 12 files into the repository root (the directory containing
`train.py`), overwriting the four modified files:

```bash
cp encoding_envelopes.py metrics_mnist.py evaluate_mnist_extended.py \
   run_referee_experiments.sh run_delta_sweep_multiseed.sh \
   run_hsv_multiseed.sh run_mnist_noise_ablation.sh \
   encoding_spectral.py config.py model.py train.py  /path/to/repo/
chmod +x /path/to/repo/run_*.sh
```

Commit before running (each run records `git rev-parse HEAD`).

## 3. What to run (and what NOT to rerun)

**No retraining of the published main results is needed.** The four-model
checkerboard ablation and the four-model MNIST comparison were already run
with 3 seeds and stand as-is. The σ = 0.1 cells of the lift-noise grid ARE
those published runs — reuse them.

Run, in priority order (times ≈ single A100):

```bash
# (1) Spectral CNN baseline + generic filter comparison    (~3 GPU-h, 9 runs)
SECTIONS="baseline filters" ./run_referee_experiments.sh

# (2) Lift-noise ablation grid, new sigma values           (~9 GPU-h, 27 runs)
SECTIONS="noise" ./run_referee_experiments.sh

# (3) Multi-seed Delta scan  (referee-required rerun)      (~4 GPU-h, 12 runs)
./run_delta_sweep_multiseed.sh

# (4) Multi-seed HSV scan    (referee-required rerun)      (~4 GPU-h, 12 runs)
./run_hsv_multiseed.sh

# (5) Extended MNIST metrics — NO retraining (see §4)      (minutes + one-time
#     ~3-minute classifier training)
python evaluate_mnist_extended.py --results_dir results_mnist_comparison_<TS> \
    --out extended_metrics_summary.json

# (6) OPTIONAL: MNIST sigma=0 spot-check                   (~2–4 GPU-days)
./run_mnist_noise_ablation.sh          # trim: MODELS="hermite"
```

Every experiment directory contains `metrics_summary.json` (mean ± std over
seeds of BV, WED_norm, JS_cell, CQS, gen_time, …), and every seed directory
contains `run_command.txt`, `git_commit.txt`, `train.log`, plus the nested run
folder with `config.json`, `final_metrics.json` and `samples/samples_final.pt`.

Seed conventions: all new experiments use seeds 1–3, matching the published
3-seed ablations. To additionally reproduce the original single-run panels,
`SEEDS="42 1 2" ./run_delta_sweep_multiseed.sh` (same for HSV).

## 4. Extended MNIST metrics without retraining

`train.py` nests outputs one level below `--output_dir`
(`seed_k/<auto_name>_<timestamp>/…`), while the original
`run_mnist_experiments.sh` deleted weights at the un-nested path
`seed_k/checkpoints/*.pt`. Those deletions therefore never matched: **the EMA
checkpoints for seeds 2 and 3 are almost certainly still on disk.** Check:

```bash
ls results_mnist_comparison_*/*/seed_*/*/checkpoints/ema_model.pt
```

* If all 12 checkpoints (4 models × 3 seeds) exist:
  `evaluate_mnist_extended.py --results_dir …` finds them automatically (it
  searches the nested layout), regenerates 10,000 samples per checkpoint with
  a fixed generator seed, and computes classifier-FID, KID (classifier +
  Inception), precision and recall per seed, then mean ± std and LaTeX rows.
  Configuration for old runs (which predate the saved `config.json`) is
  reconstructed from built-in presets that mirror `run_mnist_experiments.sh`
  verbatim (including `spectral_n_modes 28`); preset↔checkpoint mismatches
  fail loudly on the strict state-dict load.
* If some checkpoints are missing: rerun only those seeds (with the updated
  `train.py`, which also saves `samples_final.pt`), or rerun the MNIST script
  with `SAVE_ALL_WEIGHTS=true`.

Defaults: real reference = the held-out 10k MNIST test split, both real and
generated clamped to [−1, 1]; the classifier (~99% test accuracy, cached to
`mnist_classifier.pt`) is trained on the same [−1, 1] normalisation; the
Inception feature path reproduces the published `compute_fid` preprocessing
byte-for-byte so KID-Inception lives in the same feature space as the
reported FID. Quote KID as KID × 10³ (the printed table already does).

## 5. What each new experiment lets you claim

* **spectral_cnn_baseline** — identical Fourier representation, phase-space
  dimension, 10,599,176-parameter CNN, optimizer and budget; only the AdS
  envelope, warped loss and KG backbone removed. Directly answers R1(a).
* **generic_heat / generic_matern** — identical to the published "AdS" model
  except the two envelope tensors, which are replaced by a least-squares-
  matched Gaussian / Matérn-type filter sharing the radial schedule
  ξ = |k|e^{-r} (fitted parameter + residual RMS vs the AdS envelope are
  printed in `train.log` as `[ENVELOPE] …` lines and are worth quoting in the
  response letter). Directly answers R1(b) and R2's "generic multiscale
  parameterization" question. At Δ = 1.5 the AdS φ-envelope is exactly
  exp(−ξ), so the comparison is exponential vs Gaussian vs rational tails.
* **lift-noise grid** — σ ∈ {0, 0.05, 0.1 (published), 0.2} for the three
  spectral models; the Hermite rows are the most informative (endpoint slopes
  consume Π̃ directly). Answers R1's implementation-details item together
  with a manuscript paragraph stating the default (σ = 0.1, applied to Π̃
  only, in every experiment).
* **multi-seed Δ / HSV scans** — replaces the single-run scans with
  mean ± std over 3 seeds, as required; phrase the HSV conclusion as "the AdS
  model outperforms every tested HSV model".

## 6. Validation performed

Unit + integration tests run on the shipped code: AdS closed form
(Δ = 1.5 envelope ∝ e^{-ξ}, dev 4×10⁻⁸); π-envelope = ∂_r φ-envelope by
finite differences (heat, Matérn: <10⁻¹⁰); bit-equality of the "ads"
reference with the published encoder buffers; deterministic matching (lsq and
efold agree to ~10%); HSV+generic correctly rejected; default path bit-
identical to published; FID(x,x)=0 and FID(x, x+3)=D·9 exactly; KID identity
≈0/deterministic/shift-sensitive; precision-recall 1/1 on identical and 0/0
on disjoint sets; all four MNIST presets rebuild models with the paper's
exact parameter counts (13,448,514 / 13,449,668); and two full end-to-end
1-epoch training runs (heat and none envelopes) through the real `train.py`,
confirming the `[ENVELOPE]` logs, `config.json` fields, `git_commit.txt`,
`samples_final.pt`, checkerboard metrics, and that the trained checkpoint
carries the swapped envelope bit-for-bit. Not testable in the sandbox (no
dataset/weight downloads): MNIST classifier training and Inception feature
extraction — both use standard torchvision paths and run on first use of
`evaluate_mnist_extended.py`.
