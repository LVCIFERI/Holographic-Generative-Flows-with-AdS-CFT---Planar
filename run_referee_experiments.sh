#!/bin/bash
# =============================================================================
# run_referee_experiments.sh
#
# NEW experiments requested in the referee reports (checkerboard side).
# Every experiment runs N_SEEDS=3 seeds and records full provenance
# (exact command line, git commit, resolved config.json, final sample tensor).
#
# Sections (each maps to a specific referee request):
#
#   [1/4] spectral_cnn_baseline
#       Referee 1: "a spectral CNN baseline that uses the same Fourier
#       representation of the data, the same phase-space dimension, the same
#       network architecture, optimizer, and training budget, but without the
#       AdS propagator envelope, without the warped loss weighting, and
#       without the Klein-Gordon backbone."
#       -> --spectral_envelope_type none  (identity phi envelope; Pi-tilde is
#          purely ancillary lift noise, so the phase space and the 10.6M-param
#          CNN are IDENTICAL to the AdS models)
#          --backbone_scale 0.0           (no KG backbone)
#          --no-use_omega_weighting       (plain unweighted L2 loss)
#          --path_type linear
#
#   [2/4] generic_heat   and   [3/4] generic_matern
#       Referee 1 & 2: "compare the AdS spectral envelope with at least one
#       generic coarse-to-fine spectral filter (heat-kernel/Gaussian or a
#       matched Matern-type filter) implemented in the same pipeline" /
#       "what does the AdS propagator give beyond a generic multiscale
#       spectral parameterization?"
#       -> identical to the published "AdS" model (linear path, no backbone,
#          warped loss ON, lift noise 0.1) with ONLY the envelope profile
#          swapped for a least-squares-matched Gaussian / Matern filter
#          sharing the same radial schedule xi = |k| e^{-r}.
#          The fitted filter parameter and its residual RMS vs the AdS
#          envelope are printed in each train.log ("[ENVELOPE] ..." lines).
#
#   [4/4] lift-noise ablation grid
#       Referee 1: "state its default value, explain which experiments use it
#       and why, and quantify how the reported results change when the lift
#       noise is varied or set to zero."
#       -> sigma in {0.0, 0.05, 0.2} x {hermite, linear, nokg} x 3 seeds.
#          sigma = 0.1 is the published setting: those numbers already exist
#          as the 3-seed main ablation (reuse them; set INCLUDE_SIGMA_0P1=true
#          to re-run them here as well for a fully self-contained grid).
#
# Usage:
#   ./run_referee_experiments.sh                    # everything, 3 seeds
#   SECTIONS="baseline filters" ./run_referee_experiments.sh
#   SECTIONS="noise" SIGMAS="0.0" ./run_referee_experiments.sh
#   N_SEEDS=3 MATCH=lsq ./run_referee_experiments.sh
#
# Approximate cost on a single A100 (~18 min / 100-epoch checkerboard run):
#   baseline+filters : 9 runs   ~ 3 GPU-hours
#   noise grid       : 27 runs  ~ 9 GPU-hours   (36 runs ~12 h with 0.1)
# =============================================================================

set -eo pipefail

# ----------------------------------------------------------------------------
# Configuration (identical to the published run_fair_experiments.sh settings)
# ----------------------------------------------------------------------------
EPOCHS=${EPOCHS:-100}
SPECTRAL_MODES=16
HIDDEN=64
DEPTH=3
BATCH_SIZE=64
N_TRAIN=50000
N_VIZ=10000
DATASET="checkerboard"
DELTAS="1.5"
R_IR=0.0
R_UV=1.0
ODE_SOLVER="rk4"
ODE_N_STEPS=100
LIFT_NOISE_SIGMA=${LIFT_NOISE_SIGMA:-0.1}          # published default (sections 1-3)

N_SEEDS=${N_SEEDS:-3}
MATCH=${MATCH:-lsq}           # envelope matching for heat/matern: lsq | efold
SECTIONS=${SECTIONS:-"baseline filters noise"}
SIGMAS=${SIGMAS:-"0.0 0.05 0.2"}
NOISE_VARIANTS=${NOISE_VARIANTS:-"hermite linear nokg"}
INCLUDE_SIGMA_0P1=${INCLUDE_SIGMA_0P1:-false}

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR=${RESULTS_DIR:-"results_referee_${TIMESTAMP}"}
mkdir -p "$RESULTS_DIR"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
record_provenance() {
    # $1 = seed_dir ; remaining args = the exact training command
    local seed_dir=$1; shift
    mkdir -p "$seed_dir"
    printf '%q ' "$@" > "${seed_dir}/run_command.txt"
    echo "" >> "${seed_dir}/run_command.txt"
    git rev-parse HEAD > "${seed_dir}/git_commit.txt" 2>/dev/null || true
}

aggregate_seed_metrics() {
    local output_dir=$1
    local n_seeds=$2
    python3 << EOF
import json
import numpy as np
from pathlib import Path

output_dir = Path("$output_dir")
n_seeds = $n_seeds

all_metrics = []
for seed in range(1, n_seeds + 1):
    seed_dir = output_dir / f"seed_{seed}"
    # train.py nests results as <seed_dir>/<auto_name>_<timestamp>/final_metrics.json
    candidates = sorted(seed_dir.glob("final_metrics.json")) \
               + sorted(seed_dir.glob("*/final_metrics.json"))
    if candidates:
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            all_metrics.append(json.load(f))
    else:
        print(f"  Warning: no final_metrics.json under {seed_dir}")

if not all_metrics:
    print("  No metrics found to aggregate")
    raise SystemExit(0)

summary = {}
metric_keys = set()
for m in all_metrics:
    metric_keys.update(m.keys())

for key in metric_keys:
    values = [m.get(key) for m in all_metrics if key in m]
    numeric = [v for v in values
               if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if numeric:
        summary[key] = {
            "mean": float(np.mean(numeric)),
            "std": float(np.std(numeric)),
            "values": numeric,
            "n_seeds": len(numeric),
        }

with open(output_dir / "metrics_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("  Key checkerboard metrics (mean +/- std over seeds):")
for key in ["BV", "WED_norm", "JS_cell", "CQS", "gen_time"]:
    if key in summary:
        s = summary[key]
        print(f"    {key}: {s['mean']:.4f} +/- {s['std']:.4f}  (n={s['n_seeds']})")
print(f"  Saved metrics summary to {output_dir}/metrics_summary.json")
EOF
}

run_with_seeds() {
    local exp_name=$1
    local output_dir=$2
    shift 2

    echo ""
    echo "=== ${exp_name}  (${N_SEEDS} seeds) ==="
    for seed in $(seq 1 $N_SEEDS); do
        local seed_dir="${output_dir}/seed_${seed}"
        echo ""
        echo "  Seed ${seed}/${N_SEEDS} -> ${seed_dir}"
        record_provenance "$seed_dir" "$@" --output_dir "$seed_dir" \
            --name "${exp_name}_seed${seed}" --seed "$seed"
        "$@" --output_dir "$seed_dir" --name "${exp_name}_seed${seed}" \
            --seed "$seed" 2>&1 | tee "${seed_dir}/train.log"
    done
    echo ""
    echo "  Aggregating ${exp_name} ..."
    aggregate_seed_metrics "$output_dir" "$N_SEEDS"
}

# Shared flag block (identical to the published fair-comparison settings)
COMMON_FLAGS=(
    --dataset "$DATASET"
    --slice_geometry planar
    --deltas $DELTAS
    --r_ir $R_IR
    --r_uv $R_UV
    --ode_solver $ODE_SOLVER
    --ode_n_steps $ODE_N_STEPS
    --use_spectral_encoding
    --spectral_n_modes $SPECTRAL_MODES
    --residual_hidden $HIDDEN
    --residual_depth $DEPTH
    --epochs $EPOCHS
    --batch_size $BATCH_SIZE
    --n_train_samples $N_TRAIN
    --n_viz_real $N_VIZ
    --n_viz_gen $N_VIZ
    --use_ema
    --checkpoint_every 999999999
)

echo "============================================================"
echo "REFEREE EXPERIMENTS (checkerboard)"
echo "============================================================"
echo "Results:  $RESULTS_DIR"
echo "Seeds:    $N_SEEDS   Epochs: $EPOCHS"
echo "Sections: $SECTIONS"
echo "Envelope match mode: $MATCH"
echo "============================================================"

# ----------------------------------------------------------------------------
# [1/4] Spectral CNN baseline (no envelope, no warped loss, no KG backbone)
# ----------------------------------------------------------------------------
if [[ " $SECTIONS " == *" baseline "* ]]; then
    run_with_seeds "spectral_cnn_baseline" "${RESULTS_DIR}/spectral_cnn_baseline" \
        python train.py \
        "${COMMON_FLAGS[@]}" \
        --path_type linear \
        --spectral_envelope_type none \
        --backbone_scale 0.0 \
        --no-use_omega_weighting \
        --lift_noise_sigma $LIFT_NOISE_SIGMA
fi

# ----------------------------------------------------------------------------
# [2/4] Generic heat-kernel / Gaussian envelope (envelope-only swap vs "AdS")
# ----------------------------------------------------------------------------
if [[ " $SECTIONS " == *" filters "* ]]; then
    run_with_seeds "generic_heat_${MATCH}" "${RESULTS_DIR}/generic_heat_${MATCH}" \
        python train.py \
        "${COMMON_FLAGS[@]}" \
        --path_type linear \
        --spectral_envelope_type heat \
        --spectral_envelope_match $MATCH \
        --backbone_scale 0.0 \
        --lift_noise_sigma $LIFT_NOISE_SIGMA

# ----------------------------------------------------------------------------
# [3/4] Matched Matern-type envelope (envelope-only swap vs "AdS")
# ----------------------------------------------------------------------------
    run_with_seeds "generic_matern_${MATCH}" "${RESULTS_DIR}/generic_matern_${MATCH}" \
        python train.py \
        "${COMMON_FLAGS[@]}" \
        --path_type linear \
        --spectral_envelope_type matern \
        --spectral_envelope_match $MATCH \
        --backbone_scale 0.0 \
        --lift_noise_sigma $LIFT_NOISE_SIGMA
fi

# ----------------------------------------------------------------------------
# [4/4] Lift-noise ablation grid
# ----------------------------------------------------------------------------
if [[ " $SECTIONS " == *" noise "* ]]; then
    ALL_SIGMAS="$SIGMAS"
    if [[ "$INCLUDE_SIGMA_0P1" == "true" ]]; then
        ALL_SIGMAS="$ALL_SIGMAS 0.1"
    fi
    for sigma in $ALL_SIGMAS; do
        stag=$(echo "$sigma" | tr '.' 'p')
        for variant in $NOISE_VARIANTS; do
            case "$variant" in
                hermite) VARIANT_FLAGS=(--path_type hermite) ;;
                linear)  VARIANT_FLAGS=(--path_type linear) ;;
                nokg)    VARIANT_FLAGS=(--path_type linear --backbone_scale 0.0) ;;
                *) echo "Unknown noise variant: $variant"; exit 1 ;;
            esac
            run_with_seeds "lift_${variant}_sigma${stag}" \
                "${RESULTS_DIR}/lift_noise/${variant}_sigma${stag}" \
                python train.py \
                "${COMMON_FLAGS[@]}" \
                "${VARIANT_FLAGS[@]}" \
                --lift_noise_sigma $sigma
        done
    done
    echo ""
    echo "NOTE: sigma = 0.1 cells of the grid are the PUBLISHED 3-seed runs"
    echo "(run_fair_experiments.sh, N_SEEDS=3). Reuse those numbers, or set"
    echo "INCLUDE_SIGMA_0P1=true to regenerate them inside this grid."
fi

echo ""
echo "============================================================"
echo "ALL REFEREE EXPERIMENTS COMPLETED -> $RESULTS_DIR"
echo "Per-experiment: metrics_summary.json (mean +/- std over seeds)"
echo "Per-seed: run_command.txt, git_commit.txt, train.log, and the nested"
echo "run directory with config.json and samples/samples_final.pt"
echo "============================================================"
