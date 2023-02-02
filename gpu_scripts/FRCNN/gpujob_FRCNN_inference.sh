#!/usr/bin/env bash
#SBATCH --job-name=cellinr3
#SBATCH --time=24:00:00
#SBATCH --partition=staff
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --output=./gpu_scripts/logs/slurm_FRCNN_inference.out

# This activates the environment (given .envrc is in the current directory
direnv allow . && eval "\$(direnv export bash)"

python src/inference_FRCNN.py --fimgs /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/data/sequence_independent/bounding_box_processed/dt_test.txt --model R101-FPN --num_exp 1 --output-dir /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/logs/log_FRCNN/
