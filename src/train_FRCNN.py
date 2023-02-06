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

from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg, CfgNode
from detectron2.model_zoo import get_config_file, get_checkpoint_url, get_config

CLASSES = ["neurons"]
CFGS_FAST = {
    "X101-FPN": "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
    "R101-DC5": "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml",
    "R50-FPN": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    "R101-C4": "COCO-Detection/faster_rcnn_R_101_C4_3x.yaml",
    "R101-FPN": "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
}

def main(args):
    #setup_logger(output = args.log_path)
    setup_logger(output = args.output_dir + "_" + str(args.num_exp) + "/log.txt")
    print(args)
    with open(args.output_dir + "_" + str(args.num_exp) + "/metadata.txt", "w") as f:
        f.write(str(args))

    train_path = args.data_root #+ "train"
    train_s = read_imglist_txt(args.data_root + "dt_train.txt")
    val_s = read_imglist_txt(args.data_root + "dt_val.txt")
    test_s = read_imglist_txt(args.data_root + "dt_test.txt")

    partitions = ["train", "val", "test"]
    annotations = {}
    for partition in partitions:
        if os.path.isfile(train_path + partition + "_bboxes_augmented.json"):
            with open(train_path + partition + "_bboxes_augmented.json", "r") as f:
                annotations = annotations | json.load(f)
        else:
            with open(train_path + partition + "_bboxs.json", "r") as f:
                annotations = annotations | json.load(f)
    all_img = list()
    all_img.extend(train_s)

    augmentations = list()
    if args.rotations:
        rotated = list()
        for s in all_img:
            cpy_s = s[:-5]
            tmp = [cpy_s + "_" + str(i) + ".jpeg" for i in range(1,5)]
            rotated.extend(tmp)
        augmentations.extend(rotated)

    if args.cutout:
        cutout = list()
        for s in all_img:
            cpy_s = s[:-5] + "_5.jpeg"
            cutout.append(cpy_s)
        augmentations.extend(cutout)

    if args.cutmix:
        cutmix = list()
        for s in all_img:
            cpy_s = s[:-5] + "_6.jpeg"
            cutmix.append(cpy_s)
        augmentations.extend(cutmix)

    if args.mixup:
        mixup = list()
        for s in all_img:
            cpy_s = s[:-5] + "_7.jpeg"
            mixup.append(cpy_s)
        augmentations.extend(mixup)

    train, val, test = split_augmentations(augmentations, train_s, val_s, test_s)
    print("Size original set: ", len(train_s), len(val_s), len(test_s))
    print("Size of set used in current exp: ",len(train), len(val), len(test))

    for d in [["neurons_train", train], ["neurons_val", val], ["neurons_test", test]]:
        print(d[0])
        DatasetCatalog.register(d[0], lambda s=d[1]: collect_ds(s, annotations))
        global CLASSES
        MetadataCatalog.get(d[0]).set(thing_classes=CLASSES)

    global CFGS_FAST
    if not args.model in list(CFGS_FAST.keys()):
        raise ValueError("model type not supported")

    cfg_yaml = CFGS_FAST[args.model]
    print(f"Weights {cfg_yaml}")

    if args.train == True:
        cfg = config_train(ds_name_train="neurons_train",
                           ds_name_val="neurons_val",
                           base_lr=args.lr,
                           output_dir= args.output_dir + "_" + str(args.num_exp),
                           max_iter=args.max_iter,
                           model=cfg_yaml,
                           pretrained=args.pretrained)
        print(cfg.dump())
        trainer = CocoTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    if args.test == True:
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        predictor = DefaultPredictor(cfg)
        evaluator = COCOEvaluator(dataset_name="neurons_test",
                                  output_dir=cfg.OUTPUT_DIR + "/test")
        val_loader = build_detection_test_loader(cfg, "neurons_test")
        eval_results = inference_on_dataset(trainer.model, val_loader, evaluator)

        with open(cfg.OUTPUT_DIR + "/test/test_res.json", "w+") as f:
            json.dump(eval_results, f)

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "validation")

    return COCOEvaluator(dataset_name, output_dir=output_folder)

def config_train(ds_name_train: str,
                 ds_name_val: str = None,
                 output_dir: str = None,
                 base_lr: float = 0.0001,
                 max_iter: int = 15000,
                 eval_period: int = 500,
                 model: str = None,
                 pretrained: bool = False) -> CfgNode:
    cfg = get_config(model, trained=pretrained)

    cfg.DATASETS.TRAIN = (ds_name_train,)
    cfg.TEST.EVAL_PERIOD = eval_period
    cfg.DATASETS.TEST = (ds_name_val,)
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (512)
    global CLASSES
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASSES)

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = base_lr
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (2500, 5000, 10000)
    cfg.SOLVER.MAX_ITER = max_iter

    cfg.CUDNN_BENCHMARK = True
    cfg.OUTPUT_DIR = output_dir
    return cfg

def collect_ds(imgs: List[str], annotations: dict) -> List[dict]:
    l = list()
    imgs.sort()
    for im in imgs:
        im_id = os.path.basename(im)[:-5]
        ann = list()
        for bb in annotations[im_id]:
            bb = [b / 1.0 for b in bb]
            if not len(bb) == 4:
                continue
            ad = {
                "bbox": bb,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,
                "iscrowd": 0
            }
            ann.append(ad)
        height, width, _ = cv2.imread(im).shape
        d = {
            "file_name": im,
            "height": height,
            "width": width,
            "image_id": im_id,
            "annotations": ann
        }
        l.append(d)

    return l

def split_augmentations(augmented_imgs: list, train: list, val: list, test: list) -> Tuple[list, list, list]:
    if len(augmented_imgs) > 0:
        for l in augmented_imgs:
            org_id = re.sub("\_[0-9]\.jpeg$", ".jpeg",l)
            if org_id in train:
                train.append(l)
            elif org_id in val:
                val.append(l)
            elif org_id in test:
                test.append(l)

    return train, val, test

def read_imglist_txt(path: str) -> List[str]:
    with open(path, "r") as f:
        l = f.read().splitlines()
    return l

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action='store_true', help="Train mode",)
    parser.add_argument("--test", dest="test", action='store_true', help="Inference mode")
    parser.add_argument("--log_path", dest="log_path", type=str, help="Output file for logger")
    parser.add_argument("--num_exp", dest="num_exp", type=int, default=1, help="No of Experiment for output dir")
    parser.add_argument("--lr", dest="lr", type=float, default=0.01, help="Learning rate for experiment")
    parser.add_argument("--max_iter", dest="max_iter", type=int, default=750, help="Iterations overall")
    parser.add_argument("--eval_period", dest="eval_period", type=int)
    parser.add_argument("--model", dest="model", type=str, default="model")
    parser.add_argument("--add_img", dest="add_img", action="store_true")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--rotations", dest="rotations", action="store_true")
    parser.add_argument("--cutmix", dest="cutmix", action="store_true")
    parser.add_argument("--mixup", dest="mixup", action="store_true")
    parser.add_argument("--cutout", dest="cutout", action="store_true")




    parser.add_argument('--data-root', help='Root folder of data', required=True)
    parser.add_argument("--output-dir", type=str, default="")
    
    args = parser.parse_args()
    main(args)