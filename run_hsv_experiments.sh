#!/bin/bash
# =============================================================================
# run_hsv_experiments.sh
#
# Hyperscaling-Violating (HSV) Geometry Experiments
#
# This script runs comprehensive HSV geometry experiments with planar boundary,
# exploring the parameter space p ∈ [0, 1) which interpolates:
#   - p = 0: Flat (Minkowski) spacetime
#   - p → 1: AdS (Anti-de Sitter) spacetime
#
# HSV Geometries:
#   - planar_hsv: HSV with R^d boundary (full p range)
#
# Usage:
#   ./run_hsv_experiments.sh                          # Default: checkerboard, 100 epochs
#   ./run_hsv_experiments.sh checkerboard 100         # Custom dataset and epochs
#   ./run_hsv_experiments.sh gaussian_mixture 50      # Different dataset
#
# Arguments:
#   dataset (optional) - Dataset name (default: checkerboard)
#   epochs  (optional) - Number of epochs (default: 100)
#
# =============================================================================

set -e

# Parse arguments
DATASET="${1:-checkerboard}"
EPOCHS="${2:-100}"

usage() {
    echo "HSV Geometry Experiments (Planar Only)"
    echo ""
    echo "Usage: $0 [dataset] [epochs]"
    echo ""
    echo "Arguments:"
    echo "  dataset (optional) - Dataset name (default: checkerboard)"
    echo "  epochs  (optional) - Number of epochs (default: 100)"
    echo ""
    echo "Examples:"
    echo "  $0                           # Default: checkerboard, 100 epochs"
    echo "  $0 checkerboard 100          # 100 epochs on checkerboard"
    echo "  $0 gaussian_mixture 50       # 50 epochs on gaussian_mixture"
    echo ""
    echo "Available datasets:"
    echo "  checkerboard       - 2D checkerboard pattern"
    echo "  gaussian_mixture   - Mixture of Gaussians"
    echo "  swiss_roll         - Swiss roll manifold"
    echo "  two_moons          - Two moons dataset"
}

# Configuration
OUTPUT_BASE="results/hsv_$(date +%Y%m%d_%H%M%S)"
SPECTRAL_MODES=16
HIDDEN=64
DEPTH=3
BATCH_SIZE=64
N_TRAIN=50000
N_VIZ=10000
SEEDS=(42)

# HSV p values covering the full range
PLANAR_P_VALUES=(0.0 0.1 0.25 0.5 0.75 0.9)

# Utility functions
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log() {
    echo "[$(timestamp)] $1"
}

ensure_dir() {
    mkdir -p "$1"
}

run_with_seeds() {
    local base_name="$1"
    local base_dir="$2"
    shift 2
    local args=("$@")
    
    for seed in "${SEEDS[@]}"; do
        local name="${base_name}_seed${seed}"
        local output_dir="${base_dir}/seed${seed}"
        
        ensure_dir "$output_dir"
        
        log "Starting: $name"
        log "Output: $output_dir"
        
        python train.py "${args[@]}" \
            --output_dir "$output_dir" \
            --seed "$seed" \
            2>&1 | tee "${output_dir}/train.log"
        
        log "Completed: $name"
    done
}

# Main experiment runner
run_planar_hsv() {
    log "============================================"
    log "Planar HSV Experiments"
    log "============================================"
    log "Geometry: planar_hsv (valid: p ∈ [0, 1))"
    log "p=0: flat, p→1: AdS"
    
    for P in "${PLANAR_P_VALUES[@]}"; do
        EXP_NAME="planar_hsv_p${P}_${DATASET}"
        OUTPUT_DIR="${OUTPUT_BASE}/planar_hsv/${EXP_NAME}"
        
        ensure_dir "$OUTPUT_DIR"
        
        log "Running: planar_hsv, p = $P"
        
        run_with_seeds "$EXP_NAME" "$OUTPUT_DIR" \
            --dataset "$DATASET" \
            --slice_geometry planar_hsv \
            --hsv_p "$P" \
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
        
        log "Completed: planar_hsv, p = $P"
    done
}

run_baselines() {
    log "============================================"
    log "Baseline Comparisons"
    log "============================================"
    
    # Flat baseline (no AdS structure)
    log "Running flat baseline..."
    run_with_seeds "flat_${DATASET}" "${OUTPUT_BASE}/baselines/flat_${DATASET}" \
        --dataset "$DATASET" \
        --slice_geometry flat \
        --path_type hermite \
        --use_spectral_encoding \
        --spectral_n_modes $SPECTRAL_MODES \
        --cnn_hidden $HIDDEN \
        --cnn_depth $DEPTH \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --n_train $N_TRAIN \
        --n_viz_gen $N_VIZ
    
    # Planar AdS baseline
    log "Running planar AdS baseline..."
    run_with_seeds "planar_ads_${DATASET}" "${OUTPUT_BASE}/baselines/planar_ads_${DATASET}" \
        --dataset "$DATASET" \
        --slice_geometry planar \
        --path_type hermite \
        --use_spectral_encoding \
        --spectral_n_modes $SPECTRAL_MODES \
        --cnn_hidden $HIDDEN \
        --cnn_depth $DEPTH \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --n_train $N_TRAIN \
        --n_viz_gen $N_VIZ
}

# Main entry point
main() {
    log "============================================"
    log "HSV GEOMETRY EXPERIMENTS"
    log "============================================"
    log "Dataset: $DATASET"
    log "Epochs: $EPOCHS"
    log "Output: $OUTPUT_BASE"
    log ""
    
    ensure_dir "$OUTPUT_BASE"
    
    # Run planar HSV experiments
    run_planar_hsv
    
    # Run baselines
    run_baselines
    
    log "============================================"
    log "ALL HSV EXPERIMENTS COMPLETED"
    log "============================================"
    log "Results saved to: $OUTPUT_BASE"
}

# Run main
main
