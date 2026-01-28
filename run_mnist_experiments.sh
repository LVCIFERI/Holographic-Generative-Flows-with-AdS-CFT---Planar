#!/bin/bash
# =============================================================================
# run_mnist_experiments.sh
#
# Fair comparison experiments on MNIST dataset:
#   1. AdS + Hermite (full model with KG backbone + FFT encoding)
#   2. AdS + Linear (full model with KG backbone + FFT encoding, linear path)
#   3. No KG + Linear (FFT encoding, no KG backbone - ablation)
#   4. CNN Baseline (vanilla CNN flow matching, no physics)
#
# All experiments use the SAME CNN architecture (hidden=64, depth=3)
# for a fair comparison. The differences are:
#   - AdS + Hermite: FFT encoding + KG backbone, Hermite path
#   - AdS + Linear: FFT encoding + KG backbone, Linear path
#   - No KG + Linear: FFT encoding, backbone_scale=0
#   - CNN Baseline: Raw images, no FFT, no physics (--use_vanilla_cnn)
#
# Metrics computed: SWD, NLL, FID (saved to final_metrics.json)
#
# Usage:
#   ./run_mnist_experiments.sh                      # Sequential on default GPU
#   N_SEEDS=3 ./run_mnist_experiments.sh            # 3 seeds per experiment
#   PARALLEL=true ./run_mnist_experiments.sh        # Parallel on GPU 0 & 1
#   PARALLEL=true GPU0=0 GPU1=1 ./run_mnist_experiments.sh  # Custom GPU IDs
#
# Parallel mode runs:
#   GPU0: AdS + Hermite -> AdS + Linear (sequential)
#   GPU1: No KG + Linear -> CNN Baseline (sequential)
#
# =============================================================================

set -e

# =============================================================================
# Hyperparameters (matching run_image_experiments.sh for MNIST)
# =============================================================================
EPOCHS=1500
BATCH_SIZE=128
N_TRAIN=10000
N_VIZ=10000
VIZ_EVERY=50
DATASET="mnist"

# Network architecture (same for all experiments - fair comparison)
SPECTRAL_MODES=28
HIDDEN=64
DEPTH=3

# Physics parameters
DELTAS="1.5"
R_IR=0.0
R_UV=1.0

# ODE integration
ODE_SOLVER="rk4"
ODE_N_STEPS=100

# UV lift noise
LIFT_NOISE_SIGMA=0.1

# Seed configuration
N_SEEDS=${N_SEEDS:-3}
SAVE_ALL_WEIGHTS=${SAVE_ALL_WEIGHTS:-false}

# Parallel execution configuration
PARALLEL=${PARALLEL:-false}
GPU0=${GPU0:-0}
GPU1=${GPU1:-1}

# =============================================================================
# Multi-Seed Functions
# =============================================================================

aggregate_seed_metrics() {
    local output_dir=$1
    local n_seeds=$2
    
    python3 << EOF
import json
import numpy as np
from pathlib import Path

output_dir = Path("$output_dir")
n_seeds = $n_seeds

# Collect metrics from each seed
all_metrics = []
for seed in range(1, n_seeds + 1):
    seed_dir = output_dir / f"seed_{seed}"
    # Use final_metrics.json (the correct filename from train.py)
    metrics_file = seed_dir / "final_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            all_metrics.append(json.load(f))
    else:
        print(f"  Warning: {metrics_file} not found")

if not all_metrics:
    print("  No metrics found to aggregate")
    exit(0)

# Compute mean and std for each metric
summary = {}
metric_keys = set()
for m in all_metrics:
    metric_keys.update(m.keys())

# Focus on key metrics for MNIST: swd, nll, fid
key_metrics = ["swd", "nll", "fid", "gen_time"]

for key in metric_keys:
    values = [m.get(key) for m in all_metrics if key in m]
    numeric_values = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    # Filter out invalid values (-1 for FID, inf for NLL)
    if key == "fid":
        numeric_values = [v for v in numeric_values if v >= 0]
    elif key == "nll":
        numeric_values = [v for v in numeric_values if v < float('inf')]
    
    if numeric_values:
        summary[key] = {
            "mean": float(np.mean(numeric_values)),
            "std": float(np.std(numeric_values)),
            "values": numeric_values,
            "n_seeds": len(numeric_values)
        }

# Print key metrics summary
print("  Key metrics summary:")
for key in key_metrics:
    if key in summary:
        print(f"    {key}: {summary[key]['mean']:.4f} ± {summary[key]['std']:.4f}")

summary_file = output_dir / "metrics_summary.json"
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)
print(f"  Saved metrics summary to {summary_file}")
EOF
}

run_with_seeds() {
    local exp_name=$1
    local output_dir=$2
    local gpu_id=$3
    shift 3
    local train_cmd="$@"
    
    # Set CUDA device if specified
    local cuda_prefix=""
    if [[ -n "$gpu_id" ]]; then
        cuda_prefix="CUDA_VISIBLE_DEVICES=$gpu_id"
    fi
    
    if [[ $N_SEEDS -eq 1 ]]; then
        echo "Running single seed experiment on GPU $gpu_id..."
        eval $cuda_prefix $train_cmd --output_dir "$output_dir" --name "$exp_name"
    else
        echo "Running $N_SEEDS seeds on GPU $gpu_id..."
        
        for seed in $(seq 1 $N_SEEDS); do
            local seed_dir="${output_dir}/seed_${seed}"
            echo ""
            echo "  Seed $seed/$N_SEEDS -> $seed_dir"
            
            eval $cuda_prefix $train_cmd --output_dir "$seed_dir" --name "${exp_name}_seed${seed}" --seed $seed
            
            if [[ "$SAVE_ALL_WEIGHTS" == "false" && $seed -gt 1 ]]; then
                echo "  Removing weights from seed $seed to save space..."
                rm -f "${seed_dir}/checkpoints/final_model.pt" \
                      "${seed_dir}/checkpoints/best_model.pt" \
                      "${seed_dir}/checkpoints/ema_model.pt"
            fi
        done
        
        echo ""
        echo "  Aggregating metrics..."
        aggregate_seed_metrics "$output_dir" $N_SEEDS
    fi
}

# =============================================================================
# Main
# =============================================================================

# Results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_mnist_comparison_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "MNIST FAIR COMPARISON EXPERIMENTS"
echo "============================================"
echo "Results: $RESULTS_DIR"
echo "Dataset: $DATASET"
echo "Epochs: $EPOCHS"
echo "Seeds: $N_SEEDS"
echo ""
echo "Execution mode: $(if [[ "$PARALLEL" == "true" ]]; then echo "PARALLEL"; else echo "SEQUENTIAL"; fi)"
if [[ "$PARALLEL" == "true" ]]; then
    echo "  GPU $GPU0: AdS + Hermite -> CNN Baseline"
    echo "  GPU $GPU1: No KG + Linear"
fi
echo ""
echo "Physics parameters:"
echo "  Deltas: $DELTAS"
echo "  Radial: r_ir=$R_IR, r_uv=$R_UV"
echo "  ODE: $ODE_SOLVER, $ODE_N_STEPS steps"
echo "  Lift noise: $LIFT_NOISE_SIGMA"
echo ""
echo "Network (same for all - fair comparison):"
echo "  Spectral Modes: $SPECTRAL_MODES"
echo "  CNN Hidden: $HIDDEN, Depth: $DEPTH"
echo "  N_TRAIN: $N_TRAIN"
echo ""
echo "Metrics: SWD, NLL, FID"
echo ""

# Common arguments for AdS + Hermite
ADS_HERMITE_CMD="python train.py \
    --dataset $DATASET \
    --slice_geometry planar \
    --path_type hermite \
    --deltas $DELTAS \
    --r_ir $R_IR \
    --r_uv $R_UV \
    --ode_solver $ODE_SOLVER \
    --ode_n_steps $ODE_N_STEPS \
    --lift_noise_sigma $LIFT_NOISE_SIGMA \
    --use_image_encoding \
    --spectral_n_modes $SPECTRAL_MODES \
    --residual_hidden $HIDDEN \
    --residual_depth $DEPTH \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --n_train_samples $N_TRAIN \
    --n_viz_real $N_VIZ \
    --n_viz_gen $N_VIZ \
    --viz_every_epochs $VIZ_EVERY \
    --checkpoint_every 999999999 \
    --use_ema"

# Common arguments for AdS + Linear (with KG backbone)
ADS_LINEAR_CMD="python train.py \
    --dataset $DATASET \
    --slice_geometry planar \
    --path_type linear \
    --deltas $DELTAS \
    --r_ir $R_IR \
    --r_uv $R_UV \
    --ode_solver $ODE_SOLVER \
    --ode_n_steps $ODE_N_STEPS \
    --lift_noise_sigma $LIFT_NOISE_SIGMA \
    --use_image_encoding \
    --spectral_n_modes $SPECTRAL_MODES \
    --residual_hidden $HIDDEN \
    --residual_depth $DEPTH \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --n_train_samples $N_TRAIN \
    --n_viz_real $N_VIZ \
    --n_viz_gen $N_VIZ \
    --viz_every_epochs $VIZ_EVERY \
    --checkpoint_every 999999999 \
    --use_ema"

# Common arguments for No KG + Linear
NO_KG_CMD="python train.py \
    --dataset $DATASET \
    --slice_geometry planar \
    --path_type linear \
    --backbone_scale 0.0 \
    --deltas $DELTAS \
    --r_ir $R_IR \
    --r_uv $R_UV \
    --ode_solver $ODE_SOLVER \
    --ode_n_steps $ODE_N_STEPS \
    --lift_noise_sigma $LIFT_NOISE_SIGMA \
    --use_image_encoding \
    --spectral_n_modes $SPECTRAL_MODES \
    --residual_hidden $HIDDEN \
    --residual_depth $DEPTH \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --n_train_samples $N_TRAIN \
    --n_viz_real $N_VIZ \
    --n_viz_gen $N_VIZ \
    --viz_every_epochs $VIZ_EVERY \
    --checkpoint_every 999999999 \
    --use_ema"

# Common arguments for CNN Baseline + Linear (vanilla flow matching, no physics)
CNN_BASELINE_CMD="python train.py \
    --dataset $DATASET \
    --slice_geometry planar \
    --path_type linear \
    --backbone_scale 0.0 \
    --deltas $DELTAS \
    --r_ir $R_IR \
    --r_uv $R_UV \
    --ode_solver $ODE_SOLVER \
    --ode_n_steps $ODE_N_STEPS \
    --lift_noise_sigma $LIFT_NOISE_SIGMA \
    --use_vanilla_cnn \
    --residual_hidden $HIDDEN \
    --residual_depth $DEPTH \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --n_train_samples $N_TRAIN \
    --n_viz_real $N_VIZ \
    --n_viz_gen $N_VIZ \
    --viz_every_epochs $VIZ_EVERY \
    --checkpoint_every 999999999 \
    --use_ema"

# -------------------------------------------------------------------------
# Execute experiments (parallel or sequential)
# -------------------------------------------------------------------------

if [[ "$PARALLEL" == "true" ]]; then
    echo "============================================"
    echo "PARALLEL EXECUTION MODE"
    echo "============================================"
    echo ""
    
    # Create log directory
    LOG_DIR="${RESULTS_DIR}/logs"
    mkdir -p "$LOG_DIR"
    
    # -------------------------------------------------------------------------
    # Run in parallel:
    #   GPU0: AdS + Hermite -> AdS + Linear
    #   GPU1: No KG + Linear -> CNN Baseline
    # -------------------------------------------------------------------------
    echo "Running in parallel:"
    echo "  GPU $GPU0: AdS + Hermite -> AdS + Linear"
    echo "  GPU $GPU1: No KG + Linear -> CNN Baseline"
    echo ""
    
    # Start GPU0 jobs (AdS + Hermite, then AdS + Linear)
    (
        echo "[GPU $GPU0] Starting AdS + Hermite..."
        run_with_seeds "ads_hermite" "${RESULTS_DIR}/ads_hermite" "$GPU0" $ADS_HERMITE_CMD
        echo "[GPU $GPU0] Completed AdS + Hermite"
        echo ""
        echo "[GPU $GPU0] Starting AdS + Linear..."
        run_with_seeds "ads_linear" "${RESULTS_DIR}/ads_linear" "$GPU0" $ADS_LINEAR_CMD
        echo "[GPU $GPU0] Completed AdS + Linear"
    ) > "${LOG_DIR}/gpu0.log" 2>&1 &
    PID_GPU0=$!
    
    # Start GPU1 jobs (No KG + Linear, then CNN Baseline)
    (
        echo "[GPU $GPU1] Starting No KG + Linear..."
        run_with_seeds "no_kg_linear" "${RESULTS_DIR}/no_kg_linear" "$GPU1" $NO_KG_CMD
        echo "[GPU $GPU1] Completed No KG + Linear"
        echo ""
        echo "[GPU $GPU1] Starting CNN Baseline..."
        run_with_seeds "cnn_baseline" "${RESULTS_DIR}/cnn_baseline" "$GPU1" $CNN_BASELINE_CMD
        echo "[GPU $GPU1] Completed CNN Baseline"
    ) > "${LOG_DIR}/gpu1.log" 2>&1 &
    PID_GPU1=$!
    
    echo "Started background processes:"
    echo "  GPU $GPU0 (PID: $PID_GPU0) -> ${LOG_DIR}/gpu0.log"
    echo "  GPU $GPU1 (PID: $PID_GPU1) -> ${LOG_DIR}/gpu1.log"
    echo ""
    echo "Monitor progress with: tail -f ${LOG_DIR}/gpu*.log"
    echo ""
    echo "Waiting for all experiments to complete..."
    
    # Wait for both to complete
    wait $PID_GPU0
    GPU0_STATUS=$?
    wait $PID_GPU1
    GPU1_STATUS=$?
    
    echo ""
    echo "Parallel execution completed:"
    echo "  GPU $GPU0 (AdS Hermite, AdS Linear): $(if [ $GPU0_STATUS -eq 0 ]; then echo 'SUCCESS'; else echo 'FAILED'; fi)"
    echo "  GPU $GPU1 (No KG Linear, CNN Baseline): $(if [ $GPU1_STATUS -eq 0 ]; then echo 'SUCCESS'; else echo 'FAILED'; fi)"
    echo ""
    
else
    # -------------------------------------------------------------------------
    # Sequential execution (original behavior)
    # -------------------------------------------------------------------------
    
    echo "[1/4] AdS + Hermite (Full Model)"
    run_with_seeds "ads_hermite" "${RESULTS_DIR}/ads_hermite" "" $ADS_HERMITE_CMD
    echo "Completed: AdS + Hermite"
    echo ""

    echo "[2/4] AdS + Linear (Full Model with Linear Path)"
    run_with_seeds "ads_linear" "${RESULTS_DIR}/ads_linear" "" $ADS_LINEAR_CMD
    echo "Completed: AdS + Linear"
    echo ""

    echo "[3/4] No KG + Linear (Image Encoding Baseline)"
    run_with_seeds "no_kg_linear" "${RESULTS_DIR}/no_kg_linear" "" $NO_KG_CMD
    echo "Completed: No KG + Linear"
    echo ""

    echo "[4/4] CNN Baseline (Vanilla Flow Matching)"
    run_with_seeds "cnn_baseline" "${RESULTS_DIR}/cnn_baseline" "" $CNN_BASELINE_CMD
    echo "Completed: CNN Baseline"
    echo ""
fi

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
echo "============================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "============================================"
echo "Results saved to: $RESULTS_DIR"
echo "Seeds per experiment: $N_SEEDS"
echo "Execution mode: $(if [[ "$PARALLEL" == "true" ]]; then echo "PARALLEL"; else echo "SEQUENTIAL"; fi)"
echo ""
echo "Experiments:"
echo "  1. ads_hermite    - Full AdS model with Hermite path + FFT encoding"
echo "  2. ads_linear     - Full AdS model with Linear path + FFT encoding"
echo "  3. no_kg_linear   - FFT encoding, no KG backbone (ablation)"
echo "  4. cnn_baseline   - Vanilla CNN flow matching, no physics (baseline)"
echo ""
echo "Metrics saved in final_metrics.json for each experiment:"
echo "  - swd: Sliced Wasserstein Distance"
echo "  - nll: Negative Log-Likelihood (KDE-based)"
echo "  - fid: Fréchet Inception Distance"
echo ""
if [[ "$PARALLEL" == "true" ]]; then
    echo "Logs saved to:"
    echo "  ${RESULTS_DIR}/logs/gpu0.log (AdS Hermite, AdS Linear)"
    echo "  ${RESULTS_DIR}/logs/gpu1.log (No KG Linear, CNN Baseline)"
    echo ""
fi
echo "To compare results across seeds, see metrics_summary.json in each experiment folder."
