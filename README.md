# Cell Center Detection

This repository hosts the code necessary to replicate the experiments of the paper titled [TODO: Add Paper Title]. If you find the code useful or if you use it your research, please cite:
[TODO: add publication here]. The data for reproduction is available at [TODO: ADD data link].

Initial implementations by Michelle Lienhart (FCNN) and Yuliia Oksymets (FRCNN). Extended and edited by Manuel Milling. 

We provide baseline experiments for two neural network architectures here: A custom fully convolutional neural network (using PyTorch) and an implementation of a Faster RCNN, based on detectron2.

- `src/` contains all necessary code to reproduce experiments
- `gpu_scripts/` contains scripts to run on a slurm-based GPU-cluster with virtual environments
- `results/` contains results from the experiments performed for the paper
- `data/` is folder designated for the data to be unpacked to
- `logs/` is the default folder for log files  

**Data Set Preparation**

- Download the data at [TODO add link], unzip in the `data` subdirectory, i.e., an example for an existing image location should be `data/sequence_independent/data_and_cell_center_annotations/train/20190420/Images_Part_1_Adhesion/Chip03_1-000.tif` 

**FCNN (Fully Convolutional Neural Network)**

Please note that paths and model names need to be adjusted appropriately.

- In order to train the FCNN model, run the training script with custom parameters, e.g. 
`python src/train_FCNN.py --data-root data/sequence_independent/data_and_cell_center_annotations/ --log-root log/log_FCNN/ --distance-threshold 16 --peak-radius 16 --epochs 100 --lr-scheduler-steps 40`

- For evaluation, use the according script `src/compute_FCNN_scores.py`, giving the path to the trained model. The model with best performance on the validation set should be available in the directory specified with the `--log-root` parameter during training. The results for varying distance thresholds will be stored in  Run for instance:
`python src/compute_FCNN_scores.py -i data/sequence_independent/data_and_cell_center_annotations/ -m /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/log/log_FCNN/model_2023-01-25_17-06-40_nn_spec1_Gaussian=0.625_sstep_size=40_num_epochs=100_size=444_lr=0.002nopost-transformbest_validation_loss.pth --distance-threshold 2 4 8 16 32 --peak-radius 8 --objectness-score 60 -p test --result_dir /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/results/fcnn/`

- Plot the original images together with predicted and ground truth cell center predictions with `src/plot_FCNN_images.py`, e.g.,
`python src/plot_FCNN_images.py -i /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/data/sequence_independent/data_and_cell_center_annotations/test/ -m /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/log/log_FCNN/model_2023-01-25_17-06-40_nn_spec1_Gaussian=0.625_sstep_size=40_num_epochs=100_size=444_lr=0.002nopost-transformbest_validation_loss.pth -o /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/results/fcnn/images/test/`

**FRCNN (Faster Region-based Convolutional Neural Network)**

- First, prepare the data for the bounding box-based object detection task by  running python `src/bounding_box_preparation.py`. The parameter `--rescale` converts all images to a common size of 2000x2000 pixels:
`python src/bounding_box_preparation.py --data-root data/sequence_independent/data_and_cell_center_annotations/ --target-path data/sequence_independent/bounding_box_augmented/ --rescale`

- Optionally, run the offline data augmentation script.
`python src/bounding_box_data_augmentation.py --data-root data/sequence_independent/bounding_box_augmented/`

- Train the FRCNN with `src/train_FRCNN.py`. Set parameters for the index of the current experiment learning rate, number of iterations, etc. To consider augmented samples, use arguments `--rotations` `--cutmix` `--mixup` and/or `--cutout`, which have to be produced in the previous step.
`python src/train_FRCNN.py --data-root /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/data/sequence_independent/bounding_box_processed/ --model R101-FPN --train --test --num_exp 11 --eval_period 500 --max_iter 35000 --lr 0.001 --output-dir /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/logs/log_FRCNN/`

- For evaluation, first run the script for inference `src/inference_FRCNN.py` and the `src/compute_FCNN_scores.py`. The final model of the previous training is be stored in `--output-dir` and will be used. The results for varying distance thresholds will be stored in `--result_dir`. Run for instance:
`python src/inference_FRCNN.py --fimgs /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/data/sequence_independent/bounding_box_processed/dt_test.txt --model R101-FPN --num_exp 1 --output-dir /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/logs/log_FRCNN/`
`python src/compute_FRCNN_scores.py --labels /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/data/sequence_independent/bounding_box_processed/val_cell_cores.json --predictions /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/logs/log_FRCNN/_1_dt_val_predictions.json --distance-thresholds 2 4 8 16 32 --objectness-score 0.5 --result_dir /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/results/frcnn/`

- Plot the original images together with predicted and ground truth bounding boxes with `src/plot_FRCNN_images.py`, e.g.,
`python src/plot_FRCNN_images.py --label-dir /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/data/sequence_independent/bounding_box_processed/ --predictions /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/src/Yuliia/results_new/_11_dt_test_predictions.json --plot_dir /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/results/frcnn/images/test/ --input-img-dir /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/data/sequence_independent/bounding_box_processed/test/ --partition test`

- Reproduce distance-dependent plot from the paper using the `plot_results.py`:
`python src/plot_results.py --result_dir /nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/results/`

> See also:
Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, and Ross Girshick. Detectron2. https://github.com/facebookresearch/detectron2, 2019.
