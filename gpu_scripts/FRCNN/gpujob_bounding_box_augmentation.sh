#!/usr/bin/env bash
#SBATCH --job-name=cellbba
#SBATCH --time=24:00:00
#SBATCH --partition=staff
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --output=./gpu_scripts/logs/slurm_box_augmentation.out

# This activates the environment (given .envrc is in the current directory
direnv allow . && eval "\$(direnv export bash)"

python src/bounding_box_data_augmentation.py --data-root data/sequence_independent/bounding_box_augmented/
