import os
import json
import glob
import cv2
import random
import numpy as np
from typing import List
from pathlib import Path
from argparse import ArgumentParser
from evaluation import determine_hits


def collect_bb(coco_inst: list) -> dict:
    d = dict()
    for i in coco_inst:
        id = i["image_id"]
        score = round(i["score"], 3) * 100
        coords = list(map(lambda x: int(round(x)), i["bbox"]))
        if id in d.keys():
            d[id].append([coords, score])
        else:
            d[id] = [[coords, score]]
    return d


def difference(l: list) -> tuple:
    return (abs(l[0] - l[2]), abs(l[1] - l[3]))



def get_center_from_boxes(bboxes, threshold = 0):
    cell_centers = []
    box_coordinates = []
    num_boxes = len(bboxes[0])
    for i in range(num_boxes):
        confidence = bboxes[1][i]
        if confidence >= threshold:
            coordinates = bboxes[0][i]
            x_coordinate = int(coordinates[0] + (coordinates[2] - coordinates[0])/2)
            y_coordinate = int(coordinates[1] + (coordinates[3] - coordinates[1])/2)
            cell_centers.append([x_coordinate, y_coordinate])
            box_coordinates.append(coordinates)
    return cell_centers, box_coordinates


if __name__ == "__main__":
    code_path = os.path.dirname(os.path.realpath(__file__))

    # command line arg parsing
    
    parser = ArgumentParser()
    parser.add_argument('--label-dir', help='Path to labels in cell center format', required=True)
    parser.add_argument('--input-img-dir', help='Path to labels in cell center format', required=True)
    parser.add_argument('--predictions', help='Path to predictions in bounding box format', required=True)
    parser.add_argument('--distance-threshold', type=int, default=16)
    parser.add_argument('--partition', default='test')
    parser.add_argument('--objectness-score', type=float, default=0.5)
    parser.add_argument('--plot_dir', default="")
    args = parser.parse_args()
    

    # Load cell cores
    with open(args.label_dir + args.partition +"_bboxs.json") as f:
        org_bb = json.load(f)
    print(org_bb)
    with open(args.label_dir + args.partition + "_cell_cores.json") as f:
        org_cc = json.load(f)

    with open(args.predictions, "r") as f:
        predictions_all_imgs = json.load(f)



    for key in predictions_all_imgs.keys():
        im_p = args.input_img_dir + key + '.jpeg'
        im = cv2.imread(im_p)
        print(key)
        threshold = args.objectness_score
        bboxes_org = org_bb[key]
        predictions_boxes = predictions_all_imgs[key]
        
        label_centers = org_cc[key]
        predictions_centers, prediction_boxes_filtered = get_center_from_boxes(predictions_boxes, threshold=args.objectness_score)
        S_1_ext, tp, fp, fn = determine_hits(predictions_centers, label_centers, max_dist_threshold=args.distance_threshold)
        

        num_pred_boxes = len(prediction_boxes_filtered)
        for i, bbox in enumerate(bboxes_org):
            coordinates = bbox
            if len(coordinates) != 4:
                print("Problem with bbox " + str(i))
                continue
            color = (255, 0, 0)
            cv2.rectangle(im, (int(coordinates[0]), int(coordinates[1])), (int(coordinates[2]), int(coordinates[3])), color, 2)
        
        for i in range(num_pred_boxes):
            coordinates = prediction_boxes_filtered[i]
            if S_1_ext[i][2] == 1:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(im, (int(coordinates[0]), int(coordinates[1])), (int(coordinates[2]), int(coordinates[3])), color, 2)
        
        
        plot_dir = args.plot_dir
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(plot_dir + str(key) +"_predicted.jpeg", im)

