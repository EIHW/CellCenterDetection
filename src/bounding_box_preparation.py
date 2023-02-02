import os
from os import path
import cv2
import json
import glob
import argparse
#import parser
import numpy as np
from pathlib import Path

RESIZE = False
COUNT_ANNOTATED = 0
COUNT_IMG_ID = 1
IMG_IDS = dict()
NEW_IMG_SIZE = (2000, 2000)

def main(args):
    #dpaths = glob.glob("/nas/student/YuliiaOksymets/Aufnahmen_bearbeitet/train/*/*")
    Path(args.target_path).mkdir(parents=True, exist_ok=True)

    partition_dirs = ["train", "val", "test"]
    #train_path = args.target_path + "train" #"/nas/student/YuliiaOksymets/data_resized/train"
    #val_path = args.target_path + "val"
    #test_path = args.target_path + "test"
    #test_path = "/nas/student/YuliiaOksymets/data_resized/test"

    # add_path = glob.glob("/nas/student/YuliiaOksymets/additional_data/Annotiert/*/*")
    # dsn_add_path = "/nas/student/YuliiaOksymets/data_resized_add"

    # Data from Micheal
    for partition in partition_dirs:
        print("Preparing: " + partition)
        Path(args.target_path + partition + "/").mkdir(parents=True, exist_ok=True)
        global COUNT_ANNOTATED_PARTITION
        COUNT_ANNOTATED_PARTITION = 0
        global COUNT_ANNOTATED_CELLS
        COUNT_ANNOTATED_CELLS = 0
        dpaths = glob.glob(args.data_root + partition + "/*/*")
        dpaths.sort()
        part_path  = args.target_path + partition
        file_list = []
        cell_cores = dict()
        for path in dpaths:
            file_list += tiff_to_jpeg(path, part_path, train = True)
            points = create_points_dict(path)
            cell_cores.update(points)
        fname1 = os.path.join(args.target_path, partition + "_cell_cores.json")
        save_points_json(cell_cores, fname1)

        bboxs = center_to_bbox(cell_cores, 25, 20, 35, 35)
        fname2 = os.path.join(args.target_path, partition + "_bboxs.json")
        save_bbox_json(bboxs, fname2)

        fname3 = os.path.join(args.target_path, "dt_" + partition + ".txt")
        save_partition_file_list(file_list, fname3)

        # Additionally annotated data
        # add_cell_cores = dict()
        # for path in add_path:
        #     tiff_to_jpeg(path, dsn_add_path, train = True)
        #     points = create_points_dict(path)
        #     add_cell_cores.update(points)
        # fname1 = os.path.join(dsn_add_path, "cell_cores.json")
        # save_points_json(add_cell_cores, fname1)

        # bboxs = center_to_bbox(add_cell_cores, 25, 20, 35, 35)
        # fname2 = os.path.join(dsn_add_path, "bboxs.json")
        # save_bbox_json(bboxs, fname2)
        print("Annotated images: ", COUNT_ANNOTATED_PARTITION)
        print("Annotated cells: ", COUNT_ANNOTATED_CELLS)

        # dpaths = glob.glob("/nas/student/YuliiaOksymets/Aufnahmen_bearbeitet/test/*/*")
        # for path in dpaths:
        #     tiff_to_jpeg(path, test_path)

        global IMG_IDS

        #with open("/nas/student/YuliiaOksymets/data_resized/img_ids.json", "w+") as f:
        with open(args.target_path + partition +"_img_ids.json", "w+") as f:
            json.dump(IMG_IDS, f, indent=4)

def tiff_to_jpeg(src: str, dst: str, train: bool = False):
    srcs = glob.glob(path.join(src, '*.tif'))
    dsn_p_list = []
    for s in srcs:
        if train:
            if not os.path.exists(s[:-4] + ".json"):
                continue
        im = cv2.imread(s)
        global RESIZE
        RESIZE = args.rescale
        global NEW_IMG_SIZE
        if RESIZE:
            im = resize_im(im, NEW_IMG_SIZE)
        add_img_id(s)
        global COUNT_IMG_ID
        fname = str(COUNT_IMG_ID) + ".jpeg"
        update_count_id()
        dsn_p = os.path.join(dst, fname)
        dsn_p_list.append(dsn_p)
        cv2.imwrite(dsn_p, im)
    # TODO: Fuck it do global variable list here as well!
    return dsn_p_list

def add_img_id(path_img: str):
    global IMG_IDS
    global COUNT_IMG_ID
    IMG_IDS[path_img] = COUNT_IMG_ID

def update_count_id(train: bool = False, test: bool = False):
    global COUNT_IMG_ID
    COUNT_IMG_ID = COUNT_IMG_ID + 1

def resize_im(im: np.ndarray, size: tuple) -> np.ndarray:
    return cv2.resize(im, size)

def create_points_dict(img_path: str) -> dict:
    paths = glob.glob(path.join(img_path, '*.json'))
    points = {}
    for p in paths:
        fname = os.path.basename(p)[:-5]
        if fname == "cell_cores" or fname == "bboxs":
            continue
        cell_cores = get_points_from_json(p)
        global COUNT_ANNOTATED_CELLS
        COUNT_ANNOTATED_CELLS += len(cell_cores)
        id = IMG_IDS[p[:-5] + ".tif"]
        resized_ccs = resize_cell_cores(cell_cores, p[:-5] + ".tif")
        points[id] = resized_ccs
        global COUNT_ANNOTATED_PARTITION
        COUNT_ANNOTATED_PARTITION += 1
        global COUNT_ANNOTATED
        COUNT_ANNOTATED = COUNT_ANNOTATED + 1
    return points

def resize_cell_cores(cell_cores: list, im_path: str) -> list:
    im = cv2.imread(im_path)
    cpy = list()
    for cc in cell_cores:
        global RESIZE
        if RESIZE:
            global NEW_IMG_SIZE
            x_rel = cc[0] / im.shape[1]
            x_rel = int(x_rel * NEW_IMG_SIZE[1])
            y_rel = cc[1] / im.shape[0]
            y_rel = int(y_rel * NEW_IMG_SIZE[0])
            cpy.append([x_rel, y_rel])
        else:
            x_rel = cc[0] 
            y_rel = cc[1]
            cpy.append([x_rel, y_rel])
    return cpy

def center_to_bbox(points: dict, offset_x: int, offset_y: int, height: int, width: int) -> dict:
    bboxs = dict()
    for p in points.items():
        k = p[0]
        vals = p[1]
        for i, v in enumerate(vals):
            cpy_v = []
            x_new = v[0] - offset_x
            y_new = v[1] - offset_y
            if x_new < 0 or y_new < 0:
                continue
            cpy_v.append(x_new)
            cpy_v.append(y_new)
            cpy_v.append(cpy_v[0] + height)
            cpy_v.append(cpy_v[1] + width)
            vals[i] = cpy_v

        bboxs[k] = vals
    return bboxs

def save_points_json(points: dict, fpath: str):
    with open(fpath, 'w+') as f:
        json.dump(points, f)

def save_bbox_json(bboxs: dict, fpath: str):
    with open(fpath, 'w+') as f:
        json.dump(bboxs, f)

def save_partition_file_list(file_list: list, fpath: str):
    with open(fpath, 'w+') as f:
        for file_path in file_list:
            f.write(file_path + "\n")

def get_points_from_json(json_path: str) -> dict:
    with open(json_path, "r") as f:
        points = json.load(f)["cell_cores"]
    return points

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-root',
        help='Root folder of data', 
        required=True
    )
    parser.add_argument(
        '--target-path',
        help='Target Path to store processed bounding boxes', 
        default=""
    )

    parser.add_argument("--rescale", dest="rescale", action="store_true")

    args = parser.parse_args()
    

    main(args)