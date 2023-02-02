#!/usr/bin/env bash
#SBATCH --job-name=cellplo1
#SBATCH --time=24:00:00
#SBATCH --partition=staff
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --output=./gpu_scripts/logs/slurm_plot_fcnn_images.out

# This activates the environment (given .envrc is in the current directory)
direnv allow . && eval "\$(direnv export bash)"

python src/plot_FCNN_images.py -i /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/data/sequence_independent/data_and_cell_center_annotations/test/ -m /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/src/log_FCNN/model_2023-01-25_17-06-40_nn_spec1_Gaussian=0.625_sstep_size=40_num_epochs=100_size=444_lr=0.002nopost-transformbest_validation_loss.pth -o /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/results/fcnn/images/test/
