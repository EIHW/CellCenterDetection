import os
import re
import json
import glob
import cv2
import random
import argparse
#import parser
import numpy as np
from typing import List, Tuple

from torch import Tensor

from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg, CfgNode
from detectron2.model_zoo import get_config_file, get_checkpoint_url, get_config

from train_FRCNN import config_train, read_imglist_txt
from train_FRCNN import CFGS_FAST, CLASSES



def main(args):
    imgs = read_imglist_txt(args.fimgs)
    predictions = dict()

    global CFGS_FAST
    if not args.model in list(CFGS_FAST.keys()):
        raise ValueError("model type not supported")

    cfg_yaml = CFGS_FAST[args.model]
    print(f"Weights {cfg_yaml}")
    cfg = config_train(ds_name_train="neurons_train",
                           ds_name_val="neurons_val",
                           output_dir=args.output_dir + "_" + str(args.num_exp),
                           model=cfg_yaml,
                           pretrained=args.pretrained)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)
    print(cfg.dump())

    for im in imgs:
        imkey = os.path.basename(im)[:-5]
        input = cv2.imread(im)
        output = predictor(input)
        parsed_out = parse_prediction(output, imkey)
        print(output)
        print(parsed_out)
        predictions.update(parsed_out)

    partition_file_basename = os.path.splitext(os.path.basename(args.fimgs))[0]
    with open(cfg.OUTPUT_DIR + "_" + partition_file_basename + "_predictions.json", "w+") as f:
        json.dump(predictions, f)

def parse_prediction(prediction: dict, imkey: str):
    pred_boxes = prediction['instances'].get('pred_boxes').tensor.tolist()
    scores = prediction['instances'].get('scores').tolist()
    d = {imkey: [pred_boxes, scores]}
    return d

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_exp", dest="num_exp", type=int, default=1, help="No of Experiment for output dir")
    parser.add_argument("--lr", dest="lr", type=float, help="Learning rate for experiment")
    parser.add_argument("--model", dest="model", type=str)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--fimgs", dest="fimgs", type=str)
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()
    main(args)