# -*- coding: utf-8 -*-
 
import os, sys, csv
import numpy as np
from random import shuffle
# enable command line argument parsing
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
import torch.optim as optim
from skimage import io, transform, draw, color
#from skimage.draw import circle

import json


import cv2
import matplotlib.pyplot as plt
from parse import parse

#from cellcoredataset import CellCoreDataset, RandomCrop, RandomRescale, Rescale
#from network import My_Net
#from network import My_Net, find_local_maxima
#from annotation import get_files
from evaluation import determine_hits


        


def get_center_from_boxes(bboxes, threshold = 0):
    cell_centers = []
    num_boxes = len(bboxes[0])
    for i in range(num_boxes):
        confidence = bboxes[1][i]
        if confidence >= threshold:
            coordinates = bboxes[0][i]
            x_coordinate = int(coordinates[0] + (coordinates[2] - coordinates[0])/2)
            y_coordinate = int(coordinates[1] + (coordinates[3] - coordinates[1])/2)
            cell_centers.append([x_coordinate, y_coordinate])
    return cell_centers



if __name__ == "__main__":
    code_path = os.path.dirname(os.path.realpath(__file__))

    # command line arg parsing
    
    
    parser = ArgumentParser()
    #my_root_dir = os.path.join(os.path.dirname(code_path),"Aufnahmen_bearbeitet/20190420_Easter_special/Images_Part_2_DcAMP")
    parser.add_argument('--labels', help='Path to labels in cell center format', required=True)
    parser.add_argument('--predictions', help='Path to predictions in bounding box format', required=True)
    parser.add_argument('--distance-thresholds',  nargs='+', type=int, default=8)
    parser.add_argument('--objectness-score', type=float, default=0.5)
    parser.add_argument('--result_dir', default="")
    
    

    #my_root_dir = os.path.join(os.path.dirname(code_path),"Aufnahmen_bearbeitet/20190420_Easter_special/Images_Part_2_DcAMP")
    #parser.add_argument('-i', '--input_image_directory', dest='input_image_directory', type=str, required=False,
    #    help='Specify *.pth file name of a train model', 
    #    default=my_root_dir)
    #parser.add_argument('-m', '--model_filename', dest='model_filename', type=str, required=False,
    #    help='Specify *.pth file name of a train model', 
    #    default=os.path.join(os.path.dirname(code_path),"models/model_2019-06-23_14-14-56.pth"))
    #parser.add_argument('-d', '--device', dest='device_str', type=str, required=False,
    #    help='Specify *.pth file name of a train model', 
    #    default='gpu')
    #parser.add_argument('-g#', '--gpu_device_number', dest='gpu_device_number', type=int, required=False,
    #    help='specify cuda device number', 
    #    default=0)

    # Aufruf/Durchfuehrung
    args = parser.parse_args()
    with open(args.predictions) as f:
        predictions_all_imgs = json.load(f)
    with open(args.labels) as f:
        labels_all_imgs = json.load(f)
    print("Total images: " + str(len(predictions_all_imgs.keys())))
    precisions = []
    recalls = []
    F1s = []

    df_data = []

    for dist_threshold in args.distance_thresholds:
        tp_total = 0
        fp_total = 0
        fn_total = 0
        for key in predictions_all_imgs.keys():
            # predictions_all_imgs has to in bbox form
            predictions_boxes = predictions_all_imgs[key]
            
            # labels have to be in cell center form
            labels_centers = labels_all_imgs[key]
            # TODO: threshold???
            predictions_centers = get_center_from_boxes(predictions_boxes, threshold=args.objectness_score)
            S_1_ext, tp, fp, fn = determine_hits(predictions_centers, labels_centers, max_dist_threshold=dist_threshold)
            if tp != 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                F1 = 2 * (precision * recall) / (precision + recall)
            else:
                precision = 0
                recall = 0
                F1 = 0
            
            #print('Image ' + str(key) + ' precision: {:.3f}, recall: {:.3f}, F1: {:.3f}'.format(precision, recall, F1))
            tp_total += tp
            fp_total += fp
            fn_total += fn
        if tp_total != 0:
            precision = tp_total / (tp_total + fp_total)
            recall = tp_total / (tp_total + fn_total)
            F1 = 2 * (precision * recall) / (precision + recall)
        else:
            precision = 0
            recall = 0
            F1 = 0
        df_data.append([dist_threshold, precision, recall, F1])
        print("------------------------------------------------------------------------------------")
        print('Overall precision {}: {:.3f}, recall: {:.3f}, F1: {:.3f}'.format(dist_threshold ,precision, recall, F1))
        print("------------------------------------------------------------------------------------")

    if args.result_dir != "":
        Path(args.result_dir).mkdir(parents=True, exist_ok=True)
        x = args.distance_thresholds
        y = [args.distance_thresholds, precisions, recalls, F1s]
        print(len(y))
        result_file =  args.result_dir + os.path.splitext(os.path.basename(args.labels))[0] + "_frcnn.csv"
        df = pd.DataFrame(df_data, columns=["Distance Threshold", "Precision", "Recall", "F1"])
        df.to_csv(result_file, index=False)



        

