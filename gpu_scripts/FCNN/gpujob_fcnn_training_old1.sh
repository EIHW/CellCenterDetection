#!/usr/bin/env bash
#SBATCH --job-name=celldo1
#SBATCH --time=24:00:00
#SBATCH --partition=staff
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --output=./gpu_scripts/logs/slurm_fcnn_training_old1.out

# This activates the environment (given .envrc is in the current directory)
direnv allow . && eval "\$(direnv export bash)"

python src/train.py --data-root data/sequence_independent/data_and_cell_center_annotations/ --log-root src/log_FCNN/ --distance-threshold 8 --peak-radius 8 --epochs 100 --lr-scheduler-steps 40
