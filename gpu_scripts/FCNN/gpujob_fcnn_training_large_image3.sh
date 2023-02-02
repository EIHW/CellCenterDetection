#!/usr/bin/env bash
#SBATCH --job-name=celdetl3
#SBATCH --time=24:00:00
#SBATCH --partition=staff
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --output=./gpu_scripts/logs/slurm_fcnn_training_large_images_3.out

# This activates the environment (given .envrc is in the current directory)
direnv allow . && eval "\$(direnv export bash)"

python src/train_large_images.py --data-root /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/data/sequence_independent/data_and_cell_center_annotations/ --log-root /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/src/log_FCNN/ --distance-threshold 16 --peak-radius 16 --epochs 250 --lr-scheduler-steps 70 --objectness-score 60 --gauss-sigma 2 --conv-size 3
