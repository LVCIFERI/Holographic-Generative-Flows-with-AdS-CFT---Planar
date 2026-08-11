#!/bin/bash
# =============================================================================
# resume_referee_noise.sh
#
# Resume the lift-noise grid after the disk-full crash, WITHOUT redoing
# completed work, and (optionally) run the two extra control cells:
#
#   GRID (skips any seed that already has final_metrics.json):
#     sigma in {0.0, 0.05, 0.2} x {hermite, linear, nokg} x seeds 1-3
#     -> with your current tree this means: linear_sigma0p05 (3 seeds, incl.
#        redoing the crashed partial seed_1), nokg_sigma0p05 (3), and the
#        full sigma=0.2 row (9).                       ~15 runs, ~5 GPU-h
#
#   EXTRA CELLS (RUN_EXTRA_CELLS=true by default):
#     baseline_sigma0p0  - spectral CNN baseline at sigma=0
#                          (noise-matched comparison vs the sigma=0 Hermite
#                          result)                      3 runs, ~1 GPU-h
#     none_warped        - envelope 'none' WITH the warped loss at sigma=0.1
#                          (closes the loss-weighting confound)
#                                                       3 runs, ~1 GPU-h
#
# Differences vs the original run_referee_experiments.sh:
#   * --checkpoint_every 999999999 everywhere (no periodic checkpoints; this
#     is what filled the disk). End-of-run ema/best/final are still saved.
#   * Idempotent: re-running this script only executes missing seeds.
#   * Pre-flight disk check (aborts below MIN_FREE_GB unless FORCE=1).
#   * Partial run dirs (no final_metrics.json) inside a seed being rerun are
#     deleted first — they contain a corrupt checkpoint and waste space.
#
# Usage (from the repo root, same env as before):
#   ./resume_referee_noise.sh
#   RESULTS_DIR=results_referee_20260807_073359 ./resume_referee_noise.sh
#   RUN_EXTRA_CELLS=false ./resume_referee_noise.sh      # grid only
#   PRUNE_STEP_CHECKPOINTS=true ./resume_referee_noise.sh # also free space
# =============================================================================

set -eo pipefail

# ------------------------- configuration ------------------------------------
RESULTS_DIR=${RESULTS_DIR:-"results_referee_20260807_073359"}
N_SEEDS=${N_SEEDS:-3}
SIGMAS=${SIGMAS:-"0.0 0.05 0.2"}
NOISE_VARIANTS=${NOISE_VARIANTS:-"hermite linear nokg"}
RUN_EXTRA_CELLS=${RUN_EXTRA_CELLS:-true}
MIN_FREE_GB=${MIN_FREE_GB:-15}
PRUNE_STEP_CHECKPOINTS=${PRUNE_STEP_CHECKPOINTS:-false}

# Published fair-experiment settings (verbatim), plus no periodic checkpoints
EPOCHS=${EPOCHS:-100}
COMMON_FLAGS=(
    --dataset checkerboard
    --slice_geometry planar
    --deltas 1.5
    --r_ir 0.0
    --r_uv 1.0
    --ode_solver rk4
    --ode_n_steps 100
    --use_spectral_encoding
    --spectral_n_modes 16
    --residual_hidden 64
    --residual_depth 3
    --epochs $EPOCHS
    --batch_size 64
    --n_train_samples 50000
    --n_viz_real 10000
    --n_viz_gen 10000
    --use_ema
    --checkpoint_every 999999999
)

# ------------------------- helpers ------------------------------------------
seed_done() {
    # A seed is done iff a final_metrics.json exists (directly or nested).
    local seed_dir=$1
    compgen -G "${seed_dir}/final_metrics.json" > /dev/null 2>&1 && return 0
    compgen -G "${seed_dir}/*/final_metrics.json" > /dev/null 2>&1 && return 0
    return 1
}

clean_partial() {
    # Remove nested run dirs that never finished (no final_metrics.json):
    # they hold a corrupt/partial checkpoint from the crash.
    local seed_dir=$1
    [ -d "$seed_dir" ] || return 0
    for d in "$seed_dir"/*/; do
        [ -d "$d" ] || continue
        if [ ! -f "${d}final_metrics.json" ]; then
            echo "  Removing partial run dir: $d"
            rm -rf "$d"
        fi
    done
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
keys = set()
for m in all_metrics:
    keys.update(m.keys())
for key in keys:
    numeric = [m[key] for m in all_metrics
               if key in m and isinstance(m[key], (int, float))
               and not isinstance(m[key], bool)]
    if numeric:
        summary[key] = {"mean": float(np.mean(numeric)),
                        "std": float(np.std(numeric)),
                        "values": numeric, "n_seeds": len(numeric)}

with open(output_dir / "metrics_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("  Key metrics (mean +/- std over seeds):")
for key in ["BV", "WED_norm", "WED", "JS_cell", "CQS", "gen_time"]:
    if key in summary:
        s = summary[key]
        print(f"    {key}: {s['mean']:.4f} +/- {s['std']:.4f}  (n={s['n_seeds']})")
EOF
}

run_with_seeds_resumable() {
    local exp_name=$1
    local output_dir=$2
    shift 2

    echo ""
    echo "=== ${exp_name}  (${N_SEEDS} seeds, resumable) ==="
    local ran_any=false
    for seed in $(seq 1 $N_SEEDS); do
        local seed_dir="${output_dir}/seed_${seed}"
        if seed_done "$seed_dir"; then
            echo "  Seed ${seed}: already complete -> skipping"
            continue
        fi
        ran_any=true
        mkdir -p "$seed_dir"
        clean_partial "$seed_dir"
        echo ""
        echo "  Seed ${seed}/${N_SEEDS} -> ${seed_dir}"
        printf '%q ' "$@" --output_dir "$seed_dir" --name "${exp_name}_seed${seed}" \
            --seed "$seed" > "${seed_dir}/run_command.txt"
        echo "" >> "${seed_dir}/run_command.txt"
        git rev-parse HEAD > "${seed_dir}/git_commit.txt" 2>/dev/null || true
        "$@" --output_dir "$seed_dir" --name "${exp_name}_seed${seed}" \
            --seed "$seed" 2>&1 | tee "${seed_dir}/train.log"
    done
    echo ""
    echo "  Aggregating ${exp_name} ..."
    aggregate_seed_metrics "$output_dir" "$N_SEEDS"
    $ran_any || echo "  (nothing to run; summary refreshed)"
}

# ------------------------- pre-flight ---------------------------------------
if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: RESULTS_DIR '$RESULTS_DIR' not found."
    echo "Pass the existing tree, e.g.:"
    echo "  RESULTS_DIR=results_referee_20260807_073359 ./resume_referee_noise.sh"
    exit 1
fi

if [ "$PRUNE_STEP_CHECKPOINTS" = "true" ]; then
    echo "Pruning periodic step checkpoints in results_referee_* (keeping"
    echo "ema_model.pt / best_model.pt / final_model.pt) ..."
    find results_referee_* -path '*/checkpoints/*' -type f \
        ! -name 'ema_model.pt' ! -name 'best_model.pt' ! -name 'final_model.pt' \
        -print -delete | tail -3 || true
fi

FREE_GB=$(df -P "$RESULTS_DIR" | awk 'NR==2 {printf "%d", $4/1024/1024}')
echo "Free space on volume: ${FREE_GB} GB (need >= ${MIN_FREE_GB} GB)"
if [ "$FREE_GB" -lt "$MIN_FREE_GB" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "ERROR: not enough free space (this is what crashed the last run)."
    echo "Free space first — safe command (keeps only end-of-run weights):"
    echo "  find results_referee_* -path '*/checkpoints/*' -type f \\"
    echo "     ! -name 'ema_model.pt' ! -name 'best_model.pt' ! -name 'final_model.pt' -delete"
    echo "or re-run with PRUNE_STEP_CHECKPOINTS=true, or override with FORCE=1."
    exit 1
fi

echo "============================================================"
echo "RESUMING REFEREE NOISE GRID -> $RESULTS_DIR"
echo "Sigmas: $SIGMAS   Variants: $NOISE_VARIANTS   Extra cells: $RUN_EXTRA_CELLS"
echo "Completed seeds are auto-detected and skipped."
echo "============================================================"

# ------------------------- grid ---------------------------------------------
for sigma in $SIGMAS; do
    stag=$(echo "$sigma" | tr '.' 'p')
    for variant in $NOISE_VARIANTS; do
        case "$variant" in
            hermite) VARIANT_FLAGS=(--path_type hermite) ;;
            linear)  VARIANT_FLAGS=(--path_type linear) ;;
            nokg)    VARIANT_FLAGS=(--path_type linear --backbone_scale 0.0) ;;
            *) echo "Unknown variant: $variant"; exit 1 ;;
        esac
        run_with_seeds_resumable "lift_${variant}_sigma${stag}" \
            "${RESULTS_DIR}/lift_noise/${variant}_sigma${stag}" \
            python train.py \
            "${COMMON_FLAGS[@]}" \
            "${VARIANT_FLAGS[@]}" \
            --lift_noise_sigma $sigma
    done
done

# ------------------------- extra cells --------------------------------------
if [ "$RUN_EXTRA_CELLS" = "true" ]; then
    # (a) Spectral CNN baseline at sigma = 0: noise-matched partner for the
    #     sigma=0 Hermite result (baseline's Pi channel becomes exactly zero).
    run_with_seeds_resumable "spectral_cnn_baseline_sigma0p0" \
        "${RESULTS_DIR}/lift_noise/baseline_sigma0p0" \
        python train.py \
        "${COMMON_FLAGS[@]}" \
        --path_type linear \
        --spectral_envelope_type none \
        --backbone_scale 0.0 \
        --no-use_omega_weighting \
        --lift_noise_sigma 0.0

    # (b) Envelope 'none' WITH the warped loss at the published sigma = 0.1:
    #     isolates envelope removal from loss weighting (2x2 factorial cell).
    run_with_seeds_resumable "none_warped_sigma0p1" \
        "${RESULTS_DIR}/controls/none_warped_sigma0p1" \
        python train.py \
        "${COMMON_FLAGS[@]}" \
        --path_type linear \
        --spectral_envelope_type none \
        --backbone_scale 0.0 \
        --lift_noise_sigma 0.1
fi

echo ""
echo "============================================================"
echo "RESUME COMPLETE -> $RESULTS_DIR"
echo "Summaries: ${RESULTS_DIR}/lift_noise/*/metrics_summary.json"
[ "$RUN_EXTRA_CELLS" = "true" ] && \
echo "           ${RESULTS_DIR}/controls/none_warped_sigma0p1/metrics_summary.json"
echo "============================================================"
