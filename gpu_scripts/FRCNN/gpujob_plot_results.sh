#!/usr/bin/env bash
#SBATCH --job-name=cellplo2
#SBATCH --time=24:00:00
#SBATCH --partition=staff
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --output=./gpu_scripts/logs/slurm_FRCNN_plot_results.out

# This activates the environment (given .envrc is in the current directory
direnv allow . && eval "\$(direnv export bash)"

python src/plot_results.py --result_dir /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/results/
