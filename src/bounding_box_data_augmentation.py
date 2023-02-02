import cv2
import os
import re
import json
import glob
import random
import numpy as np
from typing import List, Tuple
import argparse

# TODO: not make everything in-place

def main(args):
    train_path = args.data_root
    add_path = args.data_root
    partitions = ["train", "val", "test"]
    # Data from Micheale
    for partition in partitions:
        print("Augmenting data for " + partition)
        with open(os.path.join(train_path, partition + "_" + "bboxs.json"), "r") as f:
            bboxes_org = json.load(f)

        with open(os.path.join(train_path, partition + "_" + "cell_cores.json"), "r") as f:
            cell_cores_org = json.load(f)

        imgs = glob.glob(train_path + partition + "/*[0-9].jpeg")
        imgs.sort()
        org_imgs = [f for f in imgs if re.search(r"\/[0-9]+\.jpeg$", f)]
        augm_bboxes, augm_cc = augment_main(org_imgs, bboxes_org, cell_cores_org)

        bboxes_org.update(augm_bboxes)
        with open(os.path.join(add_path, partition + "_" + "bboxes_augmented.json"), "w+") as f:
            json.dump(bboxes_org, f)

        cell_cores_org.update(augm_cc)
        with open(os.path.join(add_path, partition + "_" + "cell_cores_augmented.json"), "w+") as f:
            json.dump(cell_cores_org, f)

    # Additionaly annotated data
    # with open(os.path.join(add_path, "bboxs.json"), "r") as f:
    #     bboxes_org = json.load(f)

    # with open(os.path.join(add_path, "cell_cores.json"), "r") as f:
    #     cell_cores_org = json.load(f)

    # imgs = glob.glob(add_path + "/*[0-9].jpeg")
    # org_imgs = [f for f in imgs if re.search(r"\/[0-9]+\.jpeg$", f)]
    # augm_bboxes, augm_cc = augment_main(org_imgs, bboxes_org, cell_cores_org)

    # bboxes_org.update(augm_bboxes)
    # with open(os.path.join(add_path, "bboxes_augmented.json"), "w+") as f:
    #     json.dump(bboxes_org, f)

    # cell_cores_org.update(augm_cc)
    # with open(os.path.join(add_path, "cell_cores_augmented.json"), "w+") as f:
    #     json.dump(cell_cores_org, f)


def augment_main(org_imgs: List[str],
                 bboxes_org: dict,
                 cell_cores_org: dict) -> Tuple[dict, dict]:
    augm_bboxes = dict()
    augm_cc = dict()
    for im in org_imgs:
        img_cv = cv2.imread(im)
        bb_key = os.path.basename(im)[:-5]

        i = 1
        save_im(rotate_im(img_cv, 90), generate_path_im(im, i))
        augm_bboxes[bb_key + "_" + str(i)] = rotate_bbox(bboxes_org[bb_key], 90, img_cv.shape)
        augm_cc[bb_key + "_" + str(i)] = rotate_cell_core(cell_cores_org[bb_key], 90, img_cv.shape)

        i = i + 1
        save_im(rotate_im(img_cv, -90), generate_path_im(im, i))
        augm_bboxes[bb_key + "_" + str(i)] = rotate_bbox(bboxes_org[bb_key], -90, img_cv.shape)
        augm_cc[bb_key + "_" + str(i)] = rotate_cell_core(cell_cores_org[bb_key], -90, img_cv.shape)

        i = i + 1
        save_im(flip_im(img_cv, 180), generate_path_im(im, i))
        augm_bboxes[bb_key + "_" + str(i)] = rotate_bbox(bboxes_org[bb_key], 180, img_cv.shape)
        augm_cc[bb_key + "_" + str(i)] = rotate_cell_core(cell_cores_org[bb_key], 180, img_cv.shape)

        i = i + 1
        save_im(flip_im(img_cv, -180), generate_path_im(im, i))
        augm_bboxes[bb_key + "_" + str(i)] = rotate_bbox(bboxes_org[bb_key], -180, img_cv.shape)
        augm_cc[bb_key + "_" + str(i)] = rotate_cell_core(cell_cores_org[bb_key], -180, img_cv.shape)

        i = i + 1
        save_im(cutout(img_cv, crop_size=150, n_holes=4), generate_path_im(im, i))
        augm_bboxes[bb_key + "_" + str(i)] = bboxes_org[bb_key]
        augm_cc[bb_key + "_" + str(i)] = cell_cores_org[bb_key]

        i = i + 1
        rand_img = cv2.imread(random.choice(org_imgs))
        save_im(cutmix(img_cv, rand_img), generate_path_im(im, i))
        augm_bboxes[bb_key + "_" + str(i)] = bboxes_org[bb_key]
        augm_cc[bb_key + "_" + str(i)] = cell_cores_org[bb_key]

        i = i + 1
        rand_img = cv2.imread(random.choice(org_imgs))
        x = generate_path_im(im, i)
        save_im(mixup(img_cv, rand_img), generate_path_im(im, i))
        augm_bboxes[bb_key + "_" + str(i)] = bboxes_org[bb_key]
        augm_cc[bb_key + "_" + str(i)] = cell_cores_org[bb_key]
    return augm_bboxes, augm_cc

def save_im(im: list, path: str):
    cv2.imwrite(path, im)

def generate_path_im(org: str, counter: int) -> str:
    return org[:-5] + "_" + str(counter) + ".jpeg"

def rotate_im(im: list, angle: int) -> list:
    if angle == 90:
        return cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    if angle == -90:
        return cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)

def flip_im(im: list, direction: int) -> list:
    if direction == 180:
        return cv2.flip(im, 0)
    if direction == -180:
        return cv2.flip(im, 1)

def cutout(img: list, crop_size: int = 30, crop_value: float = 0.0, n_holes: int = 1) -> list:
    img = img / 255.0

    h, w, d = img.shape

    mask = np.ones((h, w, d))
    max_h = h - crop_size
    max_w = w - crop_size

    for _ in range(n_holes):
        x = np.random.randint(0, max_h)
        y = np.random.randint(0, max_w)
        x1, x2, y1, y2 = x, x+crop_size, y, y+crop_size
        mask[y1: y2, x1: x2, :] = crop_value

    img = img * mask

    return img * 255

def mixup(img1: list, img2: list, alpha: float=0.8, beta: float = 0.2, gamma: float = 0.0) -> list:
    img1 = img1 / 255.0
    img2 = img2 / 255.0

    dst = img1 * alpha + img2 * beta + gamma
    return dst * 255

def cutmix(img1: list, img2: list, crop_size: int = 200, crop_value: float = 0.0) -> list:
    img1 = img1 / 255.0
    img2 = img2 / 255.0

    h, w, d = img1.shape

    mask = np.ones((h, w, d))
    mask_patch = np.zeros((h, w, d))
    max_h = h - crop_size
    max_w = w - crop_size

    shape = (crop_size, crop_size, d)
    patch_shape = (0, 0, 0)
    mask_shape = (0, 0, 0)

    x_p = np.random.randint(0, max_h)
    y_p = np.random.randint(0, max_w)
    x1_p, x2_p, y1_p, y2_p = x_p, x_p+crop_size, y_p, y_p+crop_size

    patch_shape = (y2_p - y1_p, x2_p - x1_p, 3)

    x = np.random.randint(0, max_h)
    y = np.random.randint(0, max_w)
    x1, x2, y1, y2 = x, x+crop_size, y, y+crop_size

    mask_shape = (y2 - y1, x2 - x1, 3)

    patch = img2[y1_p: y2_p, x1_p: x2_p, :]
    mask[y1: y2, x1: x2, :] = crop_value
    mask_patch[y1: y2, x1: x2, :] = patch

    img1 = img1 * mask
    dst = img1 + mask_patch

    return dst * 255

def rotate_bbox(bboxs: list, angle: int, im_shape: tuple) -> list:
    cpy = list()
    w = im_shape[1]
    h = im_shape[0]
    for bb in bboxs:
        if not len(bb) == 4:
            continue
        if angle == 90:
            x_1 = w - bb[3]
            y_1 = bb[2]
            x_2 = w - bb[1]
            y_2 = bb[0]
        if angle == -90:
            x_1 = bb[3]
            y_1 = h - bb[2]
            x_2 = bb[1]
            y_2 = h - bb[0]
        if angle == 180:
            x_1 = bb[0]
            y_1 = h - bb[1]
            x_2 = bb[2]
            y_2 = h - bb[3]
        if angle == -180:
            x_1 = w - bb[0]
            y_1 = bb[1]
            x_2 = w - bb[2]
            y_2 = bb[3]
        cpy.append([x_1, y_1, x_2, y_2])
    return cpy

def rotate_cell_core(cell_cores: list, angle: int, im_shape: tuple) -> list:
    cpy = list()
    w = im_shape[1]
    h = im_shape[0]
    for cc in cell_cores:
        if angle == 90:
            x = w - cc[1]
            y = cc[0]
        if angle == -90:
            x = cc[1]
            y = h - cc[0]
        if angle == 180:
            x = cc[0]
            y = h - cc[1]
        if angle == -180:
            x = w - cc[0]
            y = cc[1]
        cpy.append([x, y])
    return cpy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-root',
        help='Root folder for data', 
        required=True
    )
    # parser.add_argument(
    #     '--target-path',
    #     help='Target folder for data', 
    #     required=True
    # )
    args = parser.parse_args()
    main(args)
