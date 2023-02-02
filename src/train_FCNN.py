# -*- coding: utf-8 -*-
import os
import numpy as np
import time
import copy
from datetime import datetime
import csv
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
import torch.optim as optim
from torch.optim import lr_scheduler


#import cv2
import matplotlib.pyplot as plt

from cellcoredataset import CellCoreDataset, RandomCrop, RandomRescale, Rescale

import cellcoredataset
from network import My_Net, init_weights, find_local_maxima
from evaluation import determine_hits#, compute_AP

def train_model(model, criterion, optimizer, scheduler, num_epochs=25, distance_threshold=8, peak_radius=8, objectness_score=60, post_transform=None):
    # Objectness Score not implemented
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000.0; best_F1_loss = 100000000.0
    best_loss_F1 = -1; best_F1 = -1
    best_epoch = -1; best_F1_epoch = -1

    if post_transform:
        print('post_transform = TRUE')
    
    running_loss_file = open('running_loss_F1.txt', "w")


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
                #scheduler.step()
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Only needed for 'val' phase
            all_detections = [] 
            hit_gt_sum = 0

            # total confusion matrix entries (validation)
            tp_total = 0
            fp_total = 0
            fn_total = 0
            if phase == 'test':
                print()
            # Iterate over data.
            for data in dataloaders[phase]:
                # Get batch data
                inputs = torch.Tensor(np.array([x['image'] for x in data]))
                labels = torch.Tensor(np.array([x['gt_map'] for x in data])).unsqueeze(1)
                # rearrange ground truth points
                batch_size = inputs.size()[0]
                gt_points = [ [] for i in range(0,batch_size) ]
                #for l in range(len(data['gt_points'])):
                #    a,b = data['gt_points'][l]
                # swapping x and y as numpy. Looks different now, maybe needs to be adapted
                
                for n in range(batch_size):
                    gt_list_instance = data[n]['gt_points']
                    for l in range(len(gt_list_instance)):
                        a,b = gt_list_instance[l] 
                        a = np.dtype('int64').type(a)
                        b = np.dtype('int64').type(b)   
                        gt_points[n].append([b, a])
                    
                """
                for l in range(len(data)):
                    a,b = data[l]['gt_points']
                
                    a = a.numpy()
                    b = b.numpy()
                    for n in range(batch_size):
                        # store row, column i.e., y, x position
                        gt_points[n].append([b[n], a[n]])
                """
                
                # if post_transform:
                #     inputs = post_transform(inputs)
                # since the images are grayscale, they have only 2 dimensions, instead of 3
                #input_size = inputs.shape
                inputs = inputs.unsqueeze(1)
                #input_size = inputs.shape

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #print(inputs.shape)
                    outputs = model(inputs)
                    #print('outputs:',outputs.shape)
                    #print('labels', labels.shape)
                    if phase != 'test':
                        loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        #scheduler.step()
                    elif (phase == 'val' or 'test'):
                        input_imgs = inputs.squeeze().detach().cpu().numpy()
                        output_imgs = outputs.squeeze().detach().cpu().numpy()
                        
                        max_locs = find_local_maxima(input_imgs, output_imgs, threshold=objectness_score, nms=peak_radius)
                        #if phase == 'test':    
                        #    print("GT points: " + str(len(gt_points[0])))
                        #    print("prediction points: " + str(len(max_locs)))
                        if len(max_locs) > 0:
                            det_list, tp, fp, fn = determine_hits(max_locs, gt_points[0], distance_threshold)
                            tp_total += tp
                            fp_total += fp
                            fn_total += fn
                            all_detections.extend(det_list)
                            hit_gt_sum += len(gt_points[0])

                # statistics
                running_loss += loss.item() #* inputs.size(0)

            # After one run through the data, update the schedular rate. 
            # Do it after one complete optimizer step. Otherwise, the first rate is skipped
            if phase == 'train':
                scheduler.step()
            else:
                # compute average prec
                R = sorted(all_detections, key=lambda x: x[2], reverse=True)
                #print(R)
                #print('len R, len hit_gt_sum', len(R), hit_gt_sum)
                N = len(R)
                rec = [ [] for i in range(0,N) ]
                prec = [ [] for i in range(0,N) ]
                hit_sum = 0
                fp_sum = 0
                
                for idx in range(len(R)):
                    i, j, score, hit, fp = R[idx]
                    hit_sum += hit
                    fp_sum += fp
                    rec[idx] = hit_sum / hit_gt_sum
                    prec[idx] = hit_sum / (idx + 1)

                #ap = compute_AP(rec, prec)
                if tp_total != 0:
                    precision = tp_total / (tp_total + fp_total)
                    recall = tp_total / (tp_total + fn_total)
                    F1 = 2 * (precision * recall) / (precision + recall)
                else:
                    precision = 0
                    recall = 0
                    F1 = 0
                tp_total = 0
                fp_total = 0
                fn_total = 0
            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                running_loss_file.write('-' * 20 + '\n')
                running_loss_file.write('Epoch: {}\nTrain loss: {}, {}, \n'.format(epoch, epoch_loss, 0))
                torch.save(model.state_dict(), args.log_root + 'current_model.pth')
            else:
                print(phase)
                print('{} Loss: {:.4f}, precision: {:.3f}, recall: {:.3f}, F1: {:.3f}'.format(phase, epoch_loss, precision, recall, F1))
                running_loss_file.write('{} loss: {}, precision: {:.3f}, recall: {:.3f}, F1: {:.3f}\n'.format(phase, epoch_loss, precision, recall, F1))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                best_loss_F1 = F1
                #best_rec_pred = [R[i]+[rec[i]]+[prec[i]] for i in range(len(R))] 
            # deep copy the model
            if phase == 'val' and F1 > best_F1:
                best_F1_loss = epoch_loss
                best_F1_model_wts = copy.deepcopy(model.state_dict())
                best_F1_epoch = epoch
                best_F1 = F1
                #best_F1_rec_pred = [R[i]+[rec[i]]+[prec[i]] for i in range(len(R))] 
        

        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    #print('Best val loss: {:8f} at epoch {}'.format(best_loss), best_epoch)

    # load best model weights
    #model.load_state_dict(best_model_wts)
    running_loss_file.write('Best val loss at epoch {}: loss:{:8f}, F1: {}\n'.format(best_epoch, best_loss, best_loss_F1))
    running_loss_file.write('Best val F1 at epoch {}: loss:{:8f}, F1: {}\n'.format(best_F1_epoch, best_F1_loss, best_F1))
    print('Best val loss at epoch {}: loss:{:8f}, F1: {}'.format(best_epoch + 1, best_loss, best_loss_F1))
    print('Best val F1 at epoch {}: loss:{:8f}, F1: {}'.format(best_F1_epoch + 1, best_F1_loss, best_F1))

    #with open('loss.txt', "w") as output:
    #    writer = csv.writer(output, lineterminator='\n')
    #    writer.writerows(best_rec_pred)
    #with open('F1.txt', "w") as output:
        
        #writer = csv.writer(output, lineterminator='\n')
        #writer.writerows(best_F1_rec_pred)


    return best_model_wts, best_F1_model_wts


import inspect
def retrieve_name(var):
        """
        Gets the name of var. Does it from the out most frame inner-wards.
        :param var: variable to get name from.
        :return: string
        """
        for fi in reversed(inspect.stack()):
            names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
            if len(names) > 0:
                return names[0]



# -----------------------------------------------------------------------------
# __main__
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # command line parameters
    parser = argparse.ArgumentParser('Cell Center Detection')
    parser.add_argument('--data-root', help='Path to features', required=True)
    parser.add_argument('--log-root', help='Path to features', default="")
    parser.add_argument('--distance-threshold', type=int, default=8)
    parser.add_argument('--peak-radius', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--lr-scheduler-steps', type=int, default=70)
    parser.add_argument('--objectness-score', type=int, default=60)
    #parser.add_argument('--gauss-peak', type=int, default=8)
    parser.add_argument("--scalehalf", dest="scalehalf", action="store_true")

    args = parser.parse_args()
    print(args)
    

    # read datasets
    cuda_no = ''
    datasets_filelist = {'train':[], 'val':[]}
    
    data_dir = args.data_root
    

    # Data augmentation and normalization for training
    # Just normalization for validation

    # TODO: Revert the transformations
    data_transforms = {
        'train': transforms.Compose([
            Rescale( int(1864 / 2) ),
            RandomRescale( 0.9, 1.1 ), 
            RandomCrop(444)
            #RandomCrop(720)
        ]),
        'val': transforms.Compose([
            Rescale( int(1864 / 2) ),
            # RandomCrop(444)
            #RandomCrop(720)
        ]),
        'test': transforms.Compose([
            Rescale( int(1864 / 2) ),
            #RandomCrop(444)
        ]),
    }

    # Make sure that normalization keeps everything in similar range
    # as without normalization.
    post_transform = transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5])
            #transforms.Normalize(mean=[127.5], std=[40.0])
        ])
    post_transform = None

    image_datasets = {x: CellCoreDataset(filenamelist=None, load_root_dir_files=True,
                        transform=data_transforms[x], my_root_dir=data_dir + x + '/', output_reduction=4)
                    for x in ['train', 'val', 'test']}
    #image_datasets['test'] = CellCoreDataset(filenamelist=datasets_filelist[x],
    #                    transform=data_transforms[x], my_root_dir=data_dir, output_reduction=4)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4 if x == 'train' else 1,
                                                shuffle=True if x == 'train' else 1, num_workers=4, collate_fn=lambda x: x )
                for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    
    device = torch.device("cuda"+cuda_no if torch.cuda.is_available() else "cpu")
    #device = "cpu"
    lr=0.002


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
    config_desc = '_nn_spec1_Gaussian=0.625_sstep_size=' + str(args.lr_scheduler_steps) + '_num_epochs=' + str(args.epochs) + '_size=444_lr=' + str(lr) +  ('' if post_transform else 'no') +'post-transform'
    
    net = My_Net(nn_spec1)
    print(net)
    print('Number of parameters: ', sum(p.numel() for p in net.parameters()))
    net.apply(init_weights)
    if device != torch.device('cpu'):
        print(device)
        net.cuda(device)
        #net.to(device)

    # loss function
    # mean-squared error between the input and the target
    criterion = nn.MSELoss(reduction='sum')
    #criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([100.0]), reduction='sum')
    criterion.to(device)

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0005)
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    # Decay LR by a factor of 0.1 every args.lr_scheduler_steps epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler_steps, gamma=0.1)

    Path(args.log_root).mkdir(parents=True, exist_ok=True)

    best_model_wts, best_F1_model_wts = train_model(net, criterion, optimizer, exp_lr_scheduler, num_epochs=args.epochs, distance_threshold=args.distance_threshold, post_transform=post_transform, objectness_score=args.objectness_score, peak_radius=args.peak_radius)

    my_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    args.lr_scheduler_steps

    
    torch.save(best_model_wts, args.log_root + 'model_' + my_time + config_desc +'_best_validation_loss.pth')
    torch.save(best_F1_model_wts, args.log_root + 'model_' + my_time + config_desc +'_best_validation_F1.pth')
    #os.rename('loss.txt', args.log_root + 'model_' + my_time + config_desc +'_loss_rec_prec.txt')
    #os.rename('F1.txt', args.log_root + 'model_' + my_time + config_desc + '_F1_rec_prec.txt')
    os.rename('running_loss_F1.txt', args.log_root + 'model_' + my_time + config_desc + '_running_loss_F1.txt')
