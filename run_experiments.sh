#!/bin/bash
# =============================================================================
# run_experiments.sh
#
# Master experiment runner for UV-stabilized generative flow matching.
#
# Document-faithful implementation of experiments for:
# - Section 2: Planar AdS slice geometry
# - Section 9: Both path types (Hermite with KG backbone, Linear without)
# - Ablation: Flat geometry (no AdS curvature)
# - HSV geometries with various p values
# - Symplectic integrators (leapfrog, implicit_midpoint)
#
# Usage:
#   ./run_experiments.sh all      # Run all experiments (toy + image)
#   ./run_experiments.sh toy      # Run only toy dataset experiments
#   ./run_experiments.sh image    # Run only image dataset experiments (MNIST)
#   ./run_experiments.sh quick    # Quick test run (1 dataset, 1 geometry, few epochs)
#   ./run_experiments.sh ablation # Run ablation studies (solvers, deltas)
#
# Estimated runtimes (on NVIDIA A100):
#   - toy:      ~4-6 hours
#   - image:    ~8-12 hours (MNIST only)
#   - all:      ~12-18 hours
#   - quick:    ~5-10 minutes
#   - ablation: ~2-4 hours
#
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Default hyperparameters (can be overridden via environment variables)
EPOCHS=${EPOCHS:-80}
SPECTRAL_MODES=16
HIDDEN=64
DEPTH=3
MLP_HIDDEN=512
MLP_DEPTH=6
BATCH_SIZE=64
N_TRAIN=50000
N_VIZ=10000

# Image-specific hyperparameters (larger models)
IMG_EPOCHS=80
IMG_MODES=28
IMG_HIDDEN=256
IMG_DEPTH=4
IMG_BATCH_SIZE=128

# Quick test hyperparameters
QUICK_EPOCHS=5
QUICK_N_TRAIN=1000
QUICK_N_VIZ=500

# Dataset arrays
TOY_DATASETS=("checkerboard" "gaussian_mixture" "swiss_roll" "two_moons")
IMAGE_DATASETS=("mnist")

# Geometry arrays (planar geometries only)
GEOMETRIES=("planar" "flat")

# HSV geometries with valid p values (planar only)
HSV_PLANAR_P_VALUES=(0.0 0.25 0.5 0.75 0.9)

# Path types
PATH_TYPES=("hermite" "linear")

# ODE Solvers
# Standard: euler, heun, midpoint, rk4
# Symplectic: leapfrog, implicit_midpoint (use with --residual_type potential)
ODE_SOLVERS=("rk4" "euler" "heun" "midpoint")
SYMPLECTIC_SOLVERS=("leapfrog" "implicit_midpoint")

# Residual types
# direct: Standard velocity residual (default)
# potential: Hamiltonian-preserving potential residual (for symplectic solvers)
RESIDUAL_TYPES=("direct" "potential")

# Laplacian type
# diagonal: FFT-based spectral Laplacian (exact for planar)
LAPLACIAN_TYPES=("diagonal")

# Conformal dimensions (delta values)
# Must satisfy Δ > d/2 = 1.0 for d=2 (unitarity bound)
# Default is (1.5, 1.5)
DELTA_VALUES=(1.5 2.0 2.5 3.0)

# Radial bounds (r_ir, r_uv)
R_IR_VALUES=(0.0 0.1 0.2 0.5)
R_UV_VALUES=(1.0 2.0 3.0 5.0)

# Combined radial bound pairs for sweep (r_ir, r_uv)
RADIAL_BOUND_PAIRS=("0.0:1.0" "0.1:2.0" "0.2:3.0" "0.0:5.0" "0.5:2.0")

# HSV u bounds (for stable training at any p)
HSV_U_UV_VALUES=(0.05 0.1 0.2)
HSV_U_IR_VALUES=(0.5 1.0 2.0)

# Path variants
PATH_HERMITE_RADIAL=false
OMEGA_WEIGHTING=true

# Output directories
RESULTS_DIR="results"
LOG_DIR="logs"

# Random seeds for statistical significance
SEEDS=(42 123 456)

# =============================================================================
# Utility Functions
# =============================================================================

timestamp() {
    date "+%Y-%m-%d_%H-%M-%S"
}

log() {
    echo "[$(timestamp)] $1"
}

ensure_dir() {
    mkdir -p "$1"
}

run_experiment() {
    local name="$1"
    local output_dir="$2"
    shift 2
    local args=("$@")
    
    ensure_dir "$output_dir"
    
    log "Starting: $name"
    log "Output: $output_dir"
    log "Args: ${args[*]}"
    
    python train.py "${args[@]}" \
        --output_dir "$output_dir" \
        2>&1 | tee "${output_dir}/train.log"
    
    log "Completed: $name"
}

run_with_seeds() {
    local base_name="$1"
    local base_dir="$2"
    shift 2
    local args=("$@")
    
    for seed in "${SEEDS[@]}"; do
        local name="${base_name}_seed${seed}"
        local output_dir="${base_dir}/seed${seed}"
        run_experiment "$name" "$output_dir" "${args[@]}" --seed "$seed"
    done
}

validate_config() {
    local geometry="$1"
    local laplacian="$2"
    local solver="$3"
    local residual="$4"
    
    # All planar geometries work with all laplacian types
    return 0
}

# =============================================================================
# Experiment Runners
# =============================================================================

run_toy_experiments() {
    local results_dir="$1"
    
    log "Running toy dataset experiments..."
    
    for dataset in "${TOY_DATASETS[@]}"; do
        for geometry in "${GEOMETRIES[@]}"; do
            for path_type in "${PATH_TYPES[@]}"; do
                local exp_name="ads_${dataset}_${path_type}_${geometry}"
                local exp_dir="${results_dir}/${exp_name}"
                
                if ! validate_config "$geometry" "diagonal" "rk4" "direct"; then
                    continue
                fi
                
                run_with_seeds "$exp_name" "$exp_dir" \
                    --dataset "$dataset" \
                    --slice_geometry "$geometry" \
                    --path_type "$path_type" \
                    --use_spectral_encoding \
                    --spectral_n_modes $SPECTRAL_MODES \
                    --cnn_hidden $HIDDEN \
                    --cnn_depth $DEPTH \
                    --epochs $EPOCHS \
                    --batch_size $BATCH_SIZE \
                    --n_train $N_TRAIN \
                    --n_viz_gen $N_VIZ
            done
        done
    done
}

run_hsv_experiments() {
    local results_dir="$1"
    
    log "Running HSV geometry experiments..."
    
    for dataset in "${TOY_DATASETS[@]}"; do
        # Planar HSV
        for p in "${HSV_PLANAR_P_VALUES[@]}"; do
            local exp_name="ads_${dataset}_hermite_planar_hsv_p${p}"
            local exp_dir="${results_dir}/${exp_name}"
            
            run_with_seeds "$exp_name" "$exp_dir" \
                --dataset "$dataset" \
                --slice_geometry planar_hsv \
                --hsv_p "$p" \
                --hsv_use_u_bounds \
                --path_type hermite \
                --use_spectral_encoding \
                --spectral_n_modes $SPECTRAL_MODES \
                --cnn_hidden $HIDDEN \
                --cnn_depth $DEPTH \
                --epochs $EPOCHS \
                --batch_size $BATCH_SIZE \
                --n_train $N_TRAIN \
                --n_viz_gen $N_VIZ
        done
    done
}

run_image_experiments() {
    local results_dir="$1"
    
    log "Running image dataset experiments..."
    
    for dataset in "${IMAGE_DATASETS[@]}"; do
        for geometry in "${GEOMETRIES[@]}"; do
            local exp_name="ads_${dataset}_hermite_${geometry}"
            local exp_dir="${results_dir}/${exp_name}"
            
            run_with_seeds "$exp_name" "$exp_dir" \
                --dataset "$dataset" \
                --slice_geometry "$geometry" \
                --path_type hermite \
                --use_spectral_encoding \
                --spectral_n_modes $IMG_MODES \
                --cnn_hidden $IMG_HIDDEN \
                --cnn_depth $IMG_DEPTH \
                --epochs $IMG_EPOCHS \
                --batch_size $IMG_BATCH_SIZE
        done
    done
}

run_ablation_studies() {
    local results_dir="$1"
    
    log "Running ablation studies..."
    
    # ODE Solver ablation
    log "Running ODE solver ablation..."
    for solver in "${ODE_SOLVERS[@]}"; do
        local exp_name="ablation_solver_${solver}"
        local exp_dir="${results_dir}/${exp_name}"
        
        run_with_seeds "$exp_name" "$exp_dir" \
            --dataset checkerboard \
            --slice_geometry planar \
            --path_type hermite \
            --ode_solver "$solver" \
            --use_spectral_encoding \
            --spectral_n_modes $SPECTRAL_MODES \
            --cnn_hidden $HIDDEN \
            --cnn_depth $DEPTH \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --n_train $N_TRAIN \
            --n_viz_gen $N_VIZ
    done
    
    # Symplectic solver ablation
    log "Running symplectic solver ablation..."
    for solver in "${SYMPLECTIC_SOLVERS[@]}"; do
        local exp_name="ablation_symplectic_${solver}"
        local exp_dir="${results_dir}/${exp_name}"
        
        run_with_seeds "$exp_name" "$exp_dir" \
            --dataset checkerboard \
            --slice_geometry planar \
            --path_type hermite \
            --ode_solver "$solver" \
            --residual_type potential \
            --use_spectral_encoding \
            --spectral_n_modes $SPECTRAL_MODES \
            --cnn_hidden $HIDDEN \
            --cnn_depth $DEPTH \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --n_train $N_TRAIN \
            --n_viz_gen $N_VIZ
    done
    
    # Delta ablation
    log "Running conformal dimension ablation..."
    for delta in "${DELTA_VALUES[@]}"; do
        local exp_name="ablation_delta_${delta}"
        local exp_dir="${results_dir}/${exp_name}"
        
        run_with_seeds "$exp_name" "$exp_dir" \
            --dataset checkerboard \
            --slice_geometry planar \
            --path_type hermite \
            --deltas "$delta" "$delta" \
            --use_spectral_encoding \
            --spectral_n_modes $SPECTRAL_MODES \
            --cnn_hidden $HIDDEN \
            --cnn_depth $DEPTH \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --n_train $N_TRAIN \
            --n_viz_gen $N_VIZ
    done
}

run_quick_test() {
    local results_dir="$1"
    
    log "Running quick test..."
    
    local exp_name="quick_test"
    local exp_dir="${results_dir}/${exp_name}"
    
    run_experiment "$exp_name" "$exp_dir" \
        --dataset checkerboard \
        --slice_geometry planar \
        --path_type hermite \
        --use_spectral_encoding \
        --spectral_n_modes 8 \
        --cnn_hidden 32 \
        --cnn_depth 2 \
        --epochs $QUICK_EPOCHS \
        --batch_size 32 \
        --n_train $QUICK_N_TRAIN \
        --n_viz_gen $QUICK_N_VIZ \
        --seed 42
    
    log "Quick test completed!"
}

# =============================================================================
# Main Entry Point
# =============================================================================

main() {
    local mode="${1:-all}"
    local results_dir="${RESULTS_DIR}/$(timestamp)"
    local log_file="${LOG_DIR}/experiments_$(timestamp).log"
    
    ensure_dir "$RESULTS_DIR"
    ensure_dir "$LOG_DIR"
    ensure_dir "$results_dir"
    
    log "Starting experiment suite: $mode"
    log "Results directory: $results_dir"
    log "Log file: $log_file"
    
    case "$mode" in
        all)
            run_toy_experiments "$results_dir/toy"
            run_hsv_experiments "$results_dir/hsv"
            run_image_experiments "$results_dir/image"
            ;;
        toy)
            run_toy_experiments "$results_dir/toy"
            ;;
        hsv)
            run_hsv_experiments "$results_dir/hsv"
            ;;
        image)
            run_image_experiments "$results_dir/image"
            ;;
        ablation)
            run_ablation_studies "$results_dir/ablation"
            ;;
        quick)
            run_quick_test "$results_dir/quick"
            ;;
        *)
            echo "Usage: $0 {all|toy|hsv|image|ablation|quick}"
            exit 1
            ;;
    esac
    
    log "Experiment suite completed: $mode"
    log "Results saved to: $results_dir"
}

main "$@"