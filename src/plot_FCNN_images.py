import os
import numpy as np
from random import shuffle
# enable command line argument parsing
from argparse import ArgumentParser
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
import torch.optim as optim
from skimage import io, transform, draw, color
from skimage.draw import circle_perimeter, disk

import json


import cv2
import matplotlib.pyplot as plt

from cellcoredataset import CellCoreDataset, RandomCrop, RandomRescale, Rescale
#from network import My_Net
from network import My_Net, find_local_maxima
import network
from annotation_tool import get_files
from evaluation import determine_hits


# def find_local_maxima(img, threshold=60, radius=8):
#     max_locs = []
#     idx = np.unravel_index(img.argmax(), img.shape)
#     while img[idx] > threshold:
#         #print(idx)
#         # convert to python list
#         max_locs.append( ( int(idx[0]), int(idx[1]) ) )
#         rr, cc = circle(idx[0], idx[1], radius, shape=img.shape)
#         img[rr, cc] = 0
#         idx = np.unravel_index(img.argmax(), img.shape)
#     return max_locs

def show_max_locs(img, max_locs, radius=5, verbose=False):
    for r,c, confidence_score in max_locs:
        rr, cc = disk((r, c), radius, shape=img.shape)
        img[rr, cc, 2] = 255
    if verbose:
        cv2.imshow('test', img)
        cv2.waitKey(1)
    return img

def show_max_locs2(img, max_locs, gt_points, det_list, radius=5, verbose=False):
    if len(gt_points) == 0:
        # We have no gt points
        # color yellow (BGR)
        for r,c, confidence_score in max_locs:
            rr, cc = disk((r, c), radius, shape=img.shape)
            img[rr, cc, 0] = 0
            img[rr, cc, 1] = 255
            img[rr, cc, 2] = 255
    else: 
        # plot all gt points in blue first (= misses)
        for r,c in gt_points:
            rr, cc = disk((r, c), radius, shape=img.shape)
            img[rr, cc, 0] = 255
            img[rr, cc, 1] = 0
            img[rr, cc, 2] = 0
            
        for r,c, confidence_score, bool_hit, bool_false_alarm in det_list:
            rr, cc = disk((r, c), radius, shape=img.shape)
            if bool_hit:
                img[rr, cc, 0] = 0
                img[rr, cc, 1] = 255
                img[rr, cc, 2] = 0
            elif bool_false_alarm:
                img[rr, cc, 0] = 0
                img[rr, cc, 1] = 0
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
    #img[:,:,0] += transform.resize(map[:,:], img.shape)
    
    # convert image from gray scale to rgb
    img = color.gray2rgb(img)

    img[:,:,0] += heat_map
    img[img >= 1.0] = 1.0

    return img


# video_filename = the complete video filename without its extension '.mp4'
def create_video(args, dataset, video_filename=None, img_output_flag=True, verbose=False, post_transform=None):
    my_all_params = []
    first_loop = True
    cap_out = None
    nr = 0
    if img_output_flag:
        os.makedirs(video_filename, exist_ok=True)

    for i, data in enumerate(dataset):
        inputs = torch.Tensor(data['image']).unsqueeze(0).unsqueeze(0)
        labels = torch.Tensor(data['gt_map']).unsqueeze(0).unsqueeze(0)
        gt_points = data['gt_points']
        #print(gt_points)

        # rearrange ground truth points
        batch_size = inputs.size()[0]
        assert batch_size == 1
        gt_points = [ ]
        for l in range(len(data['gt_points'])):
            a,b = data['gt_points'][l]
            # store row, column i.e., y, x position
            gt_points.append([b, a])


        inputs = inputs.to(device)
        outputs = net(inputs)


        input_image = inputs.squeeze().detach().cpu().numpy()
        my_output = outputs.squeeze().detach().cpu().numpy()
        my_min = np.min(my_output)
        my_max = np.max(my_output) 
        gt_map = labels.squeeze().cpu().numpy()
        
        # do only if video output is required
        if video_filename!=None and video_filename!='':
            bild1 = show_training_sample(input_image, my_output)
            if first_loop:
                first_loop = False
                # write as MP4 video
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                # write as motion jpeg video
                # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

                cap_out = cv2.VideoWriter(video_filename + '.mp4',fourcc, 25, (bild1.shape[1],bild1.shape[0]) )

            bild1 = (bild1 * 255.0).astype(dtype=np.uint8)

        image_filename1 = args.output_directory + "dnn-output_%02d_.png" % i
        image_filename2 = args.output_directory + "dnn-overlay_%02d_.png" % i
        image_filename3 = args.output_directory + "det-result_%02d_.png" % i
        # add ground truth map to image
        heat_map = transform.resize(my_output, input_image.shape) * 255.0
        heat_map[heat_map >= 255.0] = 255.0
        heat_map[heat_map <= 20.0] = 0.0
        if img_output_flag:
            cv2.imwrite(image_filename1, heat_map.astype(np.uint8))


        # find cell cores in heat map (=my_output)
        max_locs = find_local_maxima(input_image, my_output)
        det_list = []
        if len(max_locs) > 0:
            det_list, tp, fp, fn = determine_hits(max_locs, gt_points, 8**2)
        #print('#',det_list)    

        filename_anno = video_filename + ("/_%04d" % nr) + ".json"
        my_params = {}
        my_params['cell_cores'] = max_locs
        my_params['scaling'] = 2
        if img_output_flag:
            with open(filename_anno, 'w') as outfile:  
                json.dump(my_params, outfile)
        my_all_params.append(my_params)

        # and show result for debugging
        img = color.gray2rgb(input_image) * 255
        img = show_max_locs2(img.astype(np.uint8), max_locs, gt_points, det_list)
        if img_output_flag:
            cv2.imwrite(image_filename2, bild1)
            cv2.imwrite(image_filename3, img)

        if video_filename!=None and video_filename!='':
            cap_out.write(img)

        # show image result only if verbose is True
        if verbose:
            cv2.imshow('frame',bild1)
    
            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        del inputs
        del outputs
        torch.cuda.empty_cache()

        print('{}/{}: {}, {}'.format(nr, len(dataset),my_min, my_max))
        nr = nr + 1

    # Write detection into one single file
    filename_all_anno = video_filename + ".json"
    with open(filename_all_anno, 'w') as outfile:  
        json.dump(my_all_params, outfile)



if __name__ == "__main__":
    code_path = os.path.dirname(os.path.realpath(__file__))

    # command line arg parsing
    parser = ArgumentParser()
    my_root_dir = os.path.join(os.path.dirname(code_path),"Aufnahmen_bearbeitet/20190420_Easter_special/Images_Part_2_DcAMP")
    parser.add_argument('-i', '--input_image_directory', dest='input_image_directory', type=str, required=False,
        help='Specify *.pth file name of a train model', 
        default=my_root_dir)
    parser.add_argument('-m', '--model_filename', dest='model_filename', type=str, required=False,
        help='Specify *.pth file name of a train model', 
        default=os.path.join(os.path.dirname(code_path),"models/model_2019-06-23_14-14-56.pth"))
    parser.add_argument('-o', '--output_directory', dest='output_directory', type=str, required=False,
        help='Specify output directory', 
        default='-')
    parser.add_argument('--distance-threshold', type=int, default=16)
    parser.add_argument('-d', '--device', dest='device_str', type=str, required=False,
        help='Specify *.pth file name of a train model', 
        default='gpu')
    parser.add_argument('-g#', '--gpu_device_number', dest='gpu_device_number', type=int, required=False,
        help='specify cuda device number', 
        default=0)

     # Aufruf/Durchführung
    args = parser.parse_args()

    post_transform = transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    post_transform = None

    cuda_no = ':%d' % args.gpu_device_number
    device = torch.device("cuda"+cuda_no if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and args.device_str == "cpu":
        device = torch.device("cpu")

    #args.model_filename = os.path.join(os.path.dirname(code_path),"models/model_ap_2019-12-29_23-57-32_nn_spec1_Gaussian=0.625_sstep_size=70_num_epochs=250_size=444_lr=0.0005nopost-transform.pth")
    #DATASETDIR='../Aufnahmen_bearbeitet/test'
    #args.input_image_directory = DATASETDIR + '/20190420/Images_Part_3_DcAMP'
    #args.input_image_directory = '../weitere_Aufnahmen/20190118/Chip01'
    #args.model_filename = os.path.join(os.path.dirname(code_path),"models/model_ap_2019-12-29_23-57-32_nn_spec1_Gaussian=0.625_sstep_size=70_num_epochs=250_size=444_lr=0.0005nopost-transform.pth")

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

    
    _, spec_no = network.parse_model_filename(args.model_filename)
    net = My_Net(eval('nn_spec{}'.format(spec_no)))
    #net = My_Net(nn_spec1)
    net.load_state_dict(torch.load(args.model_filename, map_location=device))
    net.to(device)
    print(net)
    net.eval() 


    data_transform = transforms.Compose([
            Rescale( int(1864 / 2) )
        ])

    # load data set without annotations
    #my_root_dir = "/Users/Michelle/Documents/Augsburg_Uni/SS 2019/Bachelorarbeit/Aufnahmen_src/Aufnahmen_bearbeitet/20190420_Easter_special/Images_Part_3_DcAMP"
    #my_root_dir = os.path.join(os.path.dirname(code_path),"Aufnahmen_bearbeitet/20190420_Easter_special/Images_Part_2_DcAMP")

    # Collect test dataset
    #my_file_list = sorted( get_files( args.input_image_directory ) )
    #shuffle(my_file_list)
    # test_image_dataset = CellCoreDataset(filenamelist=my_file_list,
    #                     transform=data_transform, output_reduction=4)
    test_image_dataset = CellCoreDataset(filenamelist=None, load_root_dir_files=True,
                        my_root_dir=args.input_image_directory, transform=data_transform, output_reduction=4, sort_dir=True)
                    
    Path(args.output_directory).mkdir(parents=True, exist_ok=True)
    # create output video filename
    # video_filename == "" ==> no video output name
    # video_filename == "-" ==> create default video output name
    dataset_name = os.path.basename(args.input_image_directory)
    dataset_dirname = os.path.basename(os.path.dirname(args.input_image_directory))
    video_filename = args.output_directory + "video.mp4"
    if video_filename == "-":
        my_basename, my_ext = os.path.splitext( os.path.basename(args.model_filename) )
        video_dirname = os.path.join(
            os.path.dirname(code_path),
            "results", dataset_dirname + '_'  + dataset_name)
        os.makedirs(video_dirname, exist_ok=True)
        video_filename = os.path.join( video_dirname, my_basename) # + ".mp4")

    # video_filename = the complete video filename without its extension '.mp4'
    create_video(args, test_image_dataset, video_filename, post_transform = post_transform)

    
