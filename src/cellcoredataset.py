# -*- coding: utf-8 -*- 
import os
import json
import random
#import cv2

import torch
from skimage import io, transform, draw, color
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



# get all files (cell images, .tif files) in directories starting from root directory (my_root_dir)
# only get .tif files 
# returns list of filenames of images (.tif files)
def get_files(my_root_dir, ext_list=['.tif']):
    my_file_list = []
    for (dirpath, dirnames, filenames) in os.walk(my_root_dir):
        for filename in filenames:
            basename, ext = os.path.splitext(filename)
            if ext in ext_list:
                my_file_list.append(os.path.join(dirpath, filename))
            print(os.path.join(dirpath, filename))
    return my_file_list


# get all files (annotated cell cores, .json files) in directories starting from root directory (my_root_dir)
# only get .tif images with .json files
# returns list of filenames of images (.tif files) with existing .json files
def get_files_with_annotations(my_root_dir, ext_list=['.tif']):
    my_file_list = []
    for (dirpath, dirnames, filenames) in os.walk(my_root_dir):
        for filename in filenames:
            basename, ext = os.path.splitext(filename)
            if ext.lower() in ext_list:
                # check for associated .json file
                filename_anno = basename + '.json'
                if os.path.exists(os.path.join(dirpath, filename_anno)):
                    my_file_list.append(os.path.join(dirpath, filename))
                print(os.path.join(dirpath, filename))
    return my_file_list


# show training sample with recognized cell cores drawn in
def show_training_sample(sample):

    """Show image with recognized cell cores drawn in"""

    # convert image from gray scale to rgb
    img = color.gray2rgb(sample['image'])

    # map only with ground truth drawn in (0: nothing, 1: cell core)
    gt_map = sample['gt_map']
    # add ground truth map to image
    img[:,:,0] += gt_map[:,:]

    plt.imshow(img)
    #plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated


# -----------------------------------------------------------------------------
class RandomFlip(object):

    """RandomFlip the image in a sample.

    Args:
        flip_h_ratio (float): probabilty of horizontal flip
        flip_v_ratio (float): probabilty of vertical flip
    Returns:
        eventually flipped image and flipped ground truth.
    """

    def __init__(self, flip_h_ratio=0.5, flip_v_ratio=0.0):
        assert isinstance(flip_h_ratio, float)
        assert isinstance(flip_v_ratio, float)
        self.flip_h_ratio = flip_h_ratio
        self.flip_v_ratio = flip_v_ratio

    def __call__(self, sample):
        image, gt_points = sample['image'].astype(np.float32), sample['gt_points']

        h, w = image.shape[:2]
        img = image
        if self.flip_v_ratio < random.random():
            # copy because otherwise it is only a "view" which pytorch cannot handle
            img = np.flipud(image).copy()
            for idx in range(len(gt_points)):
                x,y = gt_points[idx]
                gt_points[idx] = [x, h-y-1]

        if self.flip_h_ratio < random.random():
            img = np.fliplr(image).copy()
            for idx in range(len(gt_points)):
                x,y = gt_points[idx]
                gt_points[idx] = [w-x-1, y]  
        return {'image': img, 'gt_points': gt_points}


# -----------------------------------------------------------------------------
class RandomWhiteNoise(object):

    """Add random white (Gaussian) noise to the image in a sample.

    Args:
        sigma (float): standard deviation of zero mean Gaussian noise
    Returns:
       noisy image and ground truth.
    """

    def __init__(self, sigma=0.01):
        assert isinstance(sigma, float)
        self.sigma = sigma

    def __call__(self, sample):
        image, gt_points = sample['image'].astype(np.float32), sample['gt_points']

        img = image + np.random.normal(0, self.sigma, image.shape )

        return {'image': img, 'gt_points': gt_points}


# -----------------------------------------------------------------------------
class Rescale(object):

    """Rescale the image in a sample to a given size.
        Int output size: Länge der kürzeren Seite
        Float output size: Factor which we multiply with hight

    Args:
        output_size (float or int): Desired output size.
        Int or float: smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    Returns:
        Rescaled image and rescaled list of ground truth.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, float))
        self.output_size = output_size

    def __call__(self, sample):
        image, gt_points = sample['image'].astype(np.float32), sample['gt_points']

        # h: hight of image
        # w: width of image
        h, w = image.shape[:2]
        
        if isinstance(self.output_size, int):
            # determine smaller edge of image
            # smaller edge == output_size
            if h > w:
                # new_w == output_size
                # new_h == output_size * h / w
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                # new_h == output_size
                # new_w == output_size * w / h
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = h * self.output_size, w * self.output_size

        # cast to int
        new_h, new_w = int(new_h), int(new_w)

        # resizes image to new size
        img = transform.resize(image, (new_h, new_w))
        # img: int (0, 255) --> float (0, 1)
        img = img.astype(np.float32) / 255.0


        # adjust resizing to list of ground truth
        # <= 0.49: round off
        # > 0.49: round up
        # rounding limit at 0.49: to avoid out of bound errors
        for i, p in enumerate(gt_points):
            x,y = p
            x = int(x * new_w / w + 0.49)
            y = int(y * new_h / h + 0.49)
            gt_points[i] = [x,y]      

        return {'image': img, 'gt_points': gt_points}


# -----------------------------------------------------------------------------
class RandomRescale(object):
    """Rescale the image in a sample to a randomly given size.

    Args:
        output_size (float or int): Desired output size.
        Int or float: smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    Returns:
        Rescaled image and rescaled list of ground truth.

    """

    # rescaling image by a random scaling factor out of (0.9, 1.1)
    def __init__(self, low=0.9, high=1.1):
        assert isinstance(low, (float))
        assert isinstance(high, (float))
        self.low = low
        self.high = high

    def __call__(self, sample):
        image, gt_points = sample['image'].astype(np.float32), sample['gt_points']

        # h: hight of image
        # w: width of image
        h, w = image.shape[:2]

        # random scaling factor between 0.9 and  1.1
        f = np.random.uniform(self.low, self.high)

        # calculate new hight and width of image
        new_h, new_w = int(h * f + 0.5), int(w * f + 0.5)

        # resizes image to new size
        img = transform.resize(image, (new_h, new_w))

        # adjust resizing to list of ground truth
        # scale x, y and cast to int
        for i, p in enumerate(gt_points):
            x,y = p
            x = int(x * f)
            y = int(y * f)
            gt_points[i] = [x,y]      

        return {'image': img, 'gt_points': gt_points}


# -----------------------------------------------------------------------------
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.

    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, gt_points = sample['image'].astype(np.float32), sample['gt_points']

        # h: hight of image
        # w: width of image
        h, w = image.shape[:2]

        # new_h: hight of cropped image
        # new_w: width of cropped image
        new_h, new_w = self.output_size

        #if (h - new_h) <= 0 or (w - new_w) <= 0:
        #    # image size too small, random crop not possible, return original image
        #    return {'image': image, 'gt_points': gt_points}

        # Random top left corner of cropped image
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        # define size of cropped image
        img = image[top: top + new_h,
                    left: left + new_w]

        # create cropped image with ground truth points
        gt_points_new = []
        for i, p in enumerate(gt_points):
            x,y = p
            x = x - left
            y = y - top
            # create new list with ground truth points from cropped image
            if x < new_w and y < new_h and x >= 0 and y >= 0:
                gt_points_new.append([x,y])      

        return {'image': img, 'gt_points': gt_points_new}


# -----------------------------------------------------------------------------
def gaussian_kernel(img_w, img_h, center_x: float, center_y: float, sigma: float):
    grid_y, grid_x = np.mgrid[0:img_h, 0:img_w]
    d2 = (grid_x -  center_x) ** 2 + (grid_y - center_y) ** 2
    return np.exp(-d2 / 2 / sigma / sigma)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class CellCoreDataset(Dataset):
    """Cell cores dataset."""

    def __init__(self, my_root_dir=None, load_root_dir_files=False, ext_list=['.tif'], filenamelist=None, transform=None, output_reduction=1, gauss_sigma=0.625, sort_dir=False):
        """
        Args:
            my_root_dir (string): Path to the .json and .tif files.
            root_dir (string): Path to the .json and .tif files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            output_reduction: Reduction of the size of output image (similar to scaling factor).
        """
        self.root_dir = my_root_dir

        self.cell_image_filenames = []
        # Achtung: es werden nur .tif Bilder hinzugefügt, die auch eine Annotation haben 
        if my_root_dir != None and load_root_dir_files:
            self.cell_image_filenames = self.cell_image_filenames + get_files_with_annotations(self.root_dir, ext_list)
            
        # Liste zu übergebener Liste addieren
        if filenamelist != None:
            self.cell_image_filenames = self.cell_image_filenames + filenamelist
        if sort_dir == True:
            self.cell_image_filenames.sort()

        self.transform = transform
        self.output_reduction = output_reduction
        self.gauss_sigma = gauss_sigma

    def __len__(self):
        return len(self.cell_image_filenames)

    def __getitem__(self, idx):
        img_filename = self.cell_image_filenames[idx]
        # read every image in grayscale
        #print(img_filename)
        ### TODO: Remove/adjust this part later on:
        # Either -4 or -3 depending on what folder hirarchy to be used
        file_name_split = img_filename.split("/")[-3:]
        #image_file = self.root_dir + os.path.join(file_name_split)
        #print(file_name_split)
        img_filename = self.root_dir + "/".join(file_name_split)
        #print(img_filename)
        
        image = io.imread(img_filename, as_gray=True)
        image_size_prior = image.shape
        # some image are uint8 with pixel values between 0 and 255 and some are
        # float with pixel values between 0.0 and 1.0. Make them all look like
        # having pixel values between 0 and 255.
        max_val = np.max(image)
        min_val = np.min(image)

        image = image.astype(np.float32) - min_val
        image = image * 255 / (max_val - min_val)


        basename, ext = os.path.splitext(img_filename)
        filename_anno = basename + '.json'
        my_params = {}
        my_params['cell_cores'] = []
        if os.path.exists(filename_anno):
            with open(filename_anno) as json_file:  
                my_params = json.load(json_file)
        # sample besteht aus Bild und Grundwahrheit
        sample = {'image': image, 'gt_points': my_params['cell_cores']}
        
        # Bilder und Annotationen transformieren
        if self.transform:
            sample = self.transform(sample)

        image_size_after = sample['image'].shape
        rescale_factor = image_size_after[0] / image_size_prior[0]
        sample['rescale_factor'] = rescale_factor
        # modify shape of image
        # default: output_reduction = 1
        new_shape = []
        for t in sample['image'].shape:
            # TODO: This here might have to be turned back, changed it because of an error with full images
            new_shape.append( int(t / self.output_reduction))
            #new_shape.append( int(t / self.output_reduction + 0.49))
        
        # convert my_params to result image
        # default: output_reduction = 1
        cell_core_map = np.zeros(tuple(new_shape), dtype=np.float32)
        for p in sample['gt_points']:
            x,y = p
            # Doing gaussian kernel here
            # if False:
            #     x = min(int(x / self.output_reduction + 0.5), new_shape[1]-1)
            #     y = min(int(y / self.output_reduction + 0.5), new_shape[0]-1)
            #     cell_core_map[y,x] = 1.0
            # else:
            x = float(x) / self.output_reduction 
            y = float(y) / self.output_reduction
            heat_map = gaussian_kernel(new_shape[1], new_shape[0], x, y, sigma = self.gauss_sigma)
            heat_map[heat_map > 1] = 1
            heat_map[heat_map < 0.01] = 0
            cell_core_map = np.maximum(cell_core_map, heat_map)
 
        #cell_core_map = cv2.GaussianBlur(cell_core_map, (5,5), 1.5)
        # get max point
        my_max = np.max(cell_core_map) + 0.001
        # normalize cell_core_map
        sample['gt_map'] = (cell_core_map / my_max).astype(np.float32)
        #print("Sample length: " + str(len(sample)))
        return sample


if __name__ == "__main__":
    print('Alive')
    my_root_dir = "/Users/Michelle/Documents/Augsburg_Uni/SS 2019/Bachelorarbeit/Aufnahmen_src/Aufnahmen_bearbeitet/20190420_Easter_special/Images_Part_1_Adhesion"
    dataset = CellCoreDataset(my_root_dir, 
                            transform = transforms.Compose(
                                [Rescale( int(1864 / 2) ),
                                #RandomRescale( 0.9, 1.1 ), 
                                RandomCrop(446)]
                            )
                )
    print(len(dataset))

    for n in range(2,100):
        print('File:', dataset.cell_image_filenames[n])
        show_training_sample(dataset[n])
        plt.pause(2)

    print(dataset[4])
