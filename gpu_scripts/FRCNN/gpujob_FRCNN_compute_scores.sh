#!/usr/bin/env bash
#SBATCH --job-name=cellcom1
#SBATCH --time=24:00:00
#SBATCH --partition=staff
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --output=./gpu_scripts/logs/slurm_FRCNN_compute_scores.out

# This activates the environment (given .envrc is in the current directory
direnv allow . && eval "\$(direnv export bash)"

python src/compute_FRCNN_scores.py --labels /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/data/sequence_independent/bounding_box_processed/val_cell_cores.json --predictions /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/logs/log_FRCNN/_1_dt_val_predictions.json --distance-thresholds 2 4 8 16 32 --objectness-score 0.5 --result_dir /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/results/frcnn/

