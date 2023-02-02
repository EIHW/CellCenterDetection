#!/usr/bin/env bash
#SBATCH --job-name=celltrr2
#SBATCH --time=24:00:00
#SBATCH --partition=staff
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --output=./gpu_scripts/logs/slurm_FRCNN_with_augment_rescaled.out

# This activates the environment (given .envrc is in the current directory
direnv allow . && eval "\$(direnv export bash)"

python src/train_FRCNN.py --data-root /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/data/sequence_independent/bounding_box_augmented/ --model R101-FPN --train --test --num_exp 7 --eval_period 500 --max_iter 30000 --lr 0.001 --output-dir /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/src/results_new/ --rotations --cutmix --mixup --cutout
