#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_experiment() {
    local script="$1"
    log "Running: $script"
    python "$SCRIPT_DIR/$script"
    log "Completed: $script"
}

log "Starting clinical experiments pipeline"

run_experiment "progressors_non_progressors_group_differences.py"
run_experiment "mci_progression_prediction_volumes.py"
run_experiment "diagnosis_classification.py"
run_experiment "simulated_clinical_trial.py"
run_experiment "spare_ad_experiment.py"
run_experiment "brain_age_gap_analysis.py"

log "All clinical experiments completed successfully"
