# -*- coding: utf-8 -*-

import os, sys, csv
import numpy as np
from random import shuffle
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
import torch.optim as optim
from skimage import io, transform, draw, color

import json


import cv2
import matplotlib.pyplot as plt
from parse import parse

from cellcoredataset import CellCoreDataset, RandomCrop, RandomRescale, Rescale
from network import My_Net, find_local_maxima
from annotation import get_files
from evaluation import determine_hits, compute_AP



def show_max_locs(img, max_locs, radius=5, verbose=False):
    for r,c in max_locs:
        rr, cc = circle(r, c, radius, shape=img.shape)
        img[rr, cc, 2] = 255
    if verbose:
        cv2.imshow('test', img)
        cv2.waitKey(1)
    return img



# show training sample with recognized cell cores drawn in
def show_training_sample(img, map):

    """Show image with recognized cell cores drawn in"""
    # add ground truth map to image
    heat_map = transform.resize(map[:,:], img.shape)
    heat_map[heat_map >= 1.0] = 1.0
    heat_map[heat_map <= 0.0] = 0.0
    
    # convert image from gray scale to rgb
    img = color.gray2rgb(img)

    img[:,:,0] += heat_map
    img[img >= 1.0] = 1.0

    return img


def compute_metrics_on_full_images(args, dataset, output_ap_filename=None, log_filename=None, dataset_type=None, verbose=False, net_choice='', post_transform=None):
    # stores all detections
    precisions = []
    recalls = []
    F1s = []
    # process each image seperately

    df_data = []
    for distance_threshold in args.distance_threshold:
        all_detections = []
        hit_gt_sum = 0
        tp_total = 0
        fp_total = 0
        fn_total = 0

        for no, data in enumerate(dataset):
            inputs = torch.Tensor(data['image']).unsqueeze(0).unsqueeze(0)
            img_shape = inputs.shape
            labels = torch.Tensor(data['gt_map']).unsqueeze(0).unsqueeze(0)
            rescale_factor = data['rescale_factor']

            # rearrange ground truth points
            batch_size = inputs.size()[0]
            gt_points = [ [] for i in range(0,batch_size) ]
            
            for n in range(batch_size):
                gt_list_instance = data['gt_points']
                for l in range(len(gt_list_instance)):
                    a,b = gt_list_instance[l] 
                    a = np.dtype('int64').type(a)
                    b = np.dtype('int64').type(b)   
                    gt_points[n].append([b, a])


            inputs = inputs.to(device)
            outputs = net(inputs)


            input_img = inputs.squeeze().detach().cpu().numpy()
            output_img = outputs.squeeze().detach().cpu().numpy()
            gt_map = labels.squeeze().cpu().numpy()
            
            # find cell cores
            max_locs = find_local_maxima(input_img, output_img, nms=args.non_maxmimum_suppression, threshold=args.objectness_score)

            if len(max_locs) > 0:
                det_list, tp, fp, fn = determine_hits(max_locs, gt_points[0], max_dist_threshold=distance_threshold * rescale_factor)
                all_detections.extend(det_list)
                hit_gt_sum += len(gt_points)
                tp_total += tp
                fp_total += fp
                fn_total += fn

                del inputs
                del outputs
                torch.cuda.empty_cache()

        if tp_total != 0:
            precision = tp_total / (tp_total + fp_total)
            recall = tp_total / (tp_total + fn_total)
            F1 = 2 * (precision * recall) / (precision + recall)
        else:
            precision = 0
            recall = 0
            F1 = 0
        
        precisions.append(precision)
        recalls.append(recall)
        F1s.append(F1)

        
        df_data.append([distance_threshold, precision, recall, F1])
        print('Distance Threshold: {}, precision: {:.3f}, recall: {:.3f}, F1: {:.3f}'.format(distance_threshold, precision, recall, F1))

        with open(log_filename, "a") as myfile:
            myfile.write('Distance Threshold: {}, precision: {:.3f}, recall: {:.3f}, F1: {:.3f}\n'.format(distance_threshold, precision, recall, F1))

    with open(log_filename, "a") as myfile:
        myfile.write('---------------------------------------------------------------------------\n')
        myfile.write('Final Evaluation')
        myfile.write('image: {}, precision: {:.3f}, recall: {:.3f}, F1: {:.3f}'.format(no, precision, recall, F1))
    
    
    return df_data
        




if __name__ == "__main__":
    code_path = os.path.dirname(os.path.realpath(__file__))

    # command line arg parsing
    
    
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_image_directory', type=str,
        help='Specify *.pth file name of a train model',  required=True)
    parser.add_argument('-m', '--model_path', type=str, required=True,
        help='Specify *.pth file name of a train model')
    parser.add_argument('-p', '--partition', type=str, required=False,
        help='Specify dataset type. Either val or test.', 
        default='test')
    parser.add_argument('-d', '--device', type=str, required=False,
        help='Specify *.pth file name of a train model', 
        default='gpu')
    parser.add_argument('-g#', '--gpu_device_number', dest='gpu_device_number', type=int, required=False,
        help='specify cuda device number', 
        default=0)
    parser.add_argument('--distance-threshold', nargs='+', type=int, default=8  )
    parser.add_argument('--non-maxmimum-suppression', type=int, default=8)
    parser.add_argument('--objectness-score', type=int, default=60)
    parser.add_argument('--result_dir', default="")

    parser.add_argument(
        '--peak-radius',
        type=int,
        default=8
    )
    parser.add_argument('-t', '--dataset_type', dest='dataset_type', type=str, required=False,
        help='Specify dataset type. Either val or test.', 
        default='test')

    args = parser.parse_args()

    data_dir = args.input_image_directory

    post_transform = transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    post_transform = None

    cuda_no = ':%d' % args.gpu_device_number
    device = torch.device("cuda"+cuda_no if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and args.device == "cpu":
        device = torch.device("cpu")
    
    nn_spec3 = [[1, 64, 2, True],
                [64, 128, 3, True],
                [128, 256, 4, False],
                [256, 512, 4, False]]
    nn_spec4 = [[1, 64, 3, True],
                [64, 128, 4, True],
                [128, 256, 5, False],
                [256, 512, 6, False]]
    nn_spec2 = [[1, 64//2, 2, True],
                [64//2, 128//2, 3, True],
                [128//2, 256//2, 4, False],
                [256//2, 512//2, 4, False]]
    nn_spec1 = [[1, 64//2, 2, True],
                [64//2, 128//2, 3, True],
                [128//2, 256//4, 4, False],
                [256//4, 512//4, 4, False]]


    data_transform = transforms.Compose([
            Rescale( int(1864 / 2) ),
            #RandomCrop(444)
        ])

    # load data set without annotations
    image_datasets = {x: CellCoreDataset(filenamelist=None, load_root_dir_files=True,
                        my_root_dir=data_dir + x + '/', transform=data_transform, output_reduction=4)
                    for x in [args.partition]}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4 if x == 'train' else 1,
                                                shuffle=False, num_workers=1, collate_fn=lambda x: x )
                for x in [args.partition]}
    dataset_sizes = {x: len(image_datasets[x]) for x in [args.partition]}

    print('Test len', dataset_sizes[args.partition])
    dataset_type = args.dataset_type

    basename, ext = os.path.splitext(args.model_path)
    
    spec_no = parse('{}_spec{}_{}', basename)[1]
    
    net = My_Net(eval('nn_spec{}'.format(spec_no)))
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.to(device)
    print(net)
    net.eval() 

    log_filename = basename + '_log_eval.txt'
    output_ap_filename = basename + '_' + '_ap_on_full_images_' + dataset_type +'.txt'
    os.makedirs(os.path.dirname(output_ap_filename), exist_ok=True)
    print('*******', output_ap_filename )
    print('*******', args.model_path )



    df_data = compute_metrics_on_full_images(args, image_datasets[args.partition], output_ap_filename, log_filename, dataset_type=dataset_type, post_transform = post_transform)
    
    if args.result_dir != "":
        Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        result_file =  args.result_dir + os.path.basename(args.partition) + "_fcnn.csv"
        df = pd.DataFrame(df_data, columns=["Distance Threshold", "Precision", "Recall", "F1"])
        df.to_csv(result_file, index=False)

    
