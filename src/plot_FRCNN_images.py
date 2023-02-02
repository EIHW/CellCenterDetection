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

# def round_coords(d: dict) -> dict:
#     new_d = dict()
#     for k in d.keys():
#         bboxes = d[k][0]
#         scores = d[k][1]
#         new_b = dict()
#         i = 1
#         for b in bboxes:
#             tmp = list(map(lambda x: int(round(x)), b))
#             new_b[i] = tmp
#             i = i + 1
#         scores_n = list(map(lambda x: round(x, 3), scores))
#         new_d[k] = [new_b, scores_n]
#     return new_d

def difference(l: list) -> tuple:
    return (abs(l[0] - l[2]), abs(l[1] - l[3]))


# def get_center_from_boxes(bboxes, threshold = 0):
#     cell_centers = []
#     num_boxes = len(bboxes[0])
#     for i in range(num_boxes):
#         confidence = bboxes[1][i]
#         if confidence >= threshold:
#             coordinates = bboxes[0][i]
#             x_coordinate = int(coordinates[0] + (coordinates[2] - coordinates[0])/2)
#             y_coordinate = int(coordinates[1] + (coordinates[3] - coordinates[1])/2)
#             cell_centers.append([x_coordinate, y_coordinate])
#     return cell_centers

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
    #my_root_dir = os.path.join(os.path.dirname(code_path),"Aufnahmen_bearbeitet/20190420_Easter_special/Images_Part_2_DcAMP")
    parser.add_argument('--label-dir', help='Path to labels in cell center format', required=True)
    parser.add_argument('--input-img-dir', help='Path to labels in cell center format', required=True)
    parser.add_argument('--predictions', help='Path to predictions in bounding box format', required=True)
    parser.add_argument('--distance-threshold', type=int, default=16)
    parser.add_argument('--partition', default='test')
    parser.add_argument('--objectness-score', type=float, default=0.5)
    parser.add_argument('--plot_dir', default="")
    args = parser.parse_args()
    
    #data_dir = args.result_dir

    # Load cell cores
    with open(args.label_dir + args.partition +"_bboxs.json") as f:
        org_bb = json.load(f)
    print(org_bb)
    with open(args.label_dir + args.partition + "_cell_cores.json") as f:
        org_cc = json.load(f)

    with open(args.predictions, "r") as f:
        predictions_all_imgs = json.load(f)



    for key in predictions_all_imgs.keys():
        #im_p = '/home/manu/eihw/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/data/sequence_independent/bounding_box_augmented/test/' + key + '.jpeg'
        im_p = args.input_img_dir + key + '.jpeg'
        im = cv2.imread(im_p)
        print(key)
        #print(im)
    #     im = cv2.resize(im, (416, 416))
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
            #confidence = bboxes[1][i]
            #if confidence >= threshold:
            coordinates = prediction_boxes_filtered[i]
            # not necessary now
            #x_coordinate_center = int(coordinates[0] + (coordinates[2] - coordinates[0])/2)
            #y_coordinate_center = int(coordinates[1] + (coordinates[3] - coordinates[1])/2)
            if S_1_ext[i][2] == 1:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(im, (int(coordinates[0]), int(coordinates[1])), (int(coordinates[2]), int(coordinates[3])), color, 2)
                #cv2.putText(im, "%.1f" % score + "%", (int(coordinates[0]), int(coordinates[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        
            #b = box[0]
            #score = box[1]
            
            
            #cv2.putText(im, "%.1f" % score + "%", (int(b[0]), int(b[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        #for b in bb_org:
        #    cv2.rectangle(im, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
        plot_dir = args.plot_dir
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(plot_dir + str(key) +"_predicted.jpeg", im)

