#!/usr/bin/env bash
#SBATCH --job-name=cellplo1
#SBATCH --time=24:00:00
#SBATCH --partition=staff
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --output=./gpu_scripts/logs/slurm_FRCNN_plot_images.out

# This activates the environment (given .envrc is in the current directory
direnv allow . && eval "\$(direnv export bash)"

python src/plot_FRCNN_images.py --label-dir /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/data/sequence_independent/bounding_box_processed/ --predictions /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/src/Yuliia/results_new/_11_dt_test_predictions.json --plot_dir /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/results/frcnn/images/test/ --input-img-dir /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/data/sequence_independent/bounding_box_processed/test/ --partition test
