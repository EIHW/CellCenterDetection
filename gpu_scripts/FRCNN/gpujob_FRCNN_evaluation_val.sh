#!/usr/bin/env bash
#SBATCH --job-name=cellevr2
#SBATCH --time=24:00:00
#SBATCH --partition=staff
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --output=./gpu_scripts/logs/slurm_FRCNN_evaluation_no_augment_no_rescaled_val.out

# This activates the environment (given .envrc is in the current directory
direnv allow . && eval "\$(direnv export bash)"

python src/compute_FRCNN_scores.py --labels data/sequence_independent/bounding_box_processed/val_cell_cores.json --predictions src/Yuliia/results_new/_11_dt_val_predictions.json --distance-thresholds 2 4 8 16 32 --objectness-score  0.5 --result_dir results/frcnn/
