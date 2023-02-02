# -*- coding: utf-8 -*-

import collections, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
import torch.optim as optim

import numpy as np
#import cv2
import matplotlib.pyplot as plt
from skimage import transform
# changed
#from skimage.draw import circle
from skimage.draw import circle_perimeter, disk
from parse import parse

from cellcoredataset import CellCoreDataset, RandomCrop, RandomRescale, Rescale

# -----------------------------------------------------------------------------
# parse moodel filename
# -----------------------------------------------------------------------------

def parse_model_filename(model_filename):
    basename, ext = os.path.splitext(model_filename)
    net_choice = ''
    if basename.find('model_loss') != -1:
        basename = basename.replace("model_loss", "model")
        net_choice = 'loss'
    elif basename.find('model_ap') != -1:
        basename = basename.replace("model_ap", "model")
        net_choice = 'ap'

    spec_no = parse('{}_spec{}_{}', basename)[1]
    return [net_choice, spec_no]

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def get_local_maxima(img, threshold=60, radius=8):
    max_locs = []
    idx = np.unravel_index(img.argmax(), img.shape)
    counter = 0
    while img[idx] > threshold:
        #print(counter)
        counter += 1
        #print(np.max(img))
        #print(idx)
        # convert to python list
        max_locs.append( [ int(idx[0]), int(idx[1]), img[idx].item() ] )
        rr, cc = disk((idx[0], idx[1]), radius, shape=img.shape)
        #rr, cc = circle_perimeter(idx[0], idx[1], radius, method='andres', shape=img.shape)
        img[rr, cc] = 0
        idx = np.unravel_index(img.argmax(), img.shape)
    return max_locs

def find_local_maxima(input_img, output_map, threshold=60, nms=8):
    heat_map = transform.resize(output_map, input_img.shape) * 255.0
    # find cell cores
    max_locs = get_local_maxima(heat_map, threshold, nms)
#    if len(max_locs) > 100:
#        plt.imshow(heat_map)
#        plt.savefig("/home/manu/eihw/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/src/Michelle/debug/" + str(len(max_locs)) + ".png")
    return max_locs    

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def add_layer_level(layers: collections.OrderedDict, num, layer_spec, conv_filter_size=3):
    num_inputs, num_outputs, num_convs, do_pooling = layer_spec
    layers['conv{}_{}'.format(num,0) ] = nn.Conv2d(num_inputs, num_outputs, conv_filter_size, padding=int(conv_filter_size/2))
    layers['relu{}_{}'.format(num,0) ] = nn.ReLU(inplace=True) 
    for n in range(1,num_convs):
        layers['conv{}_{}'.format(num,n) ] = nn.Conv2d(num_outputs, num_outputs, conv_filter_size, padding=int(conv_filter_size/2))
        # layers['bn2d{}_{}'.format(num,n) ] = nn.BatchNorm2d(num_outputs)
        layers['relu{}_{}'.format(num,n) ] = nn.ReLU(inplace=True)
    if do_pooling:
        layers['max_pool_2d{}_{}'.format(num,n) ] = nn.MaxPool2d((2, 2))


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class My_Net(nn.Module):

    def __init__(
            self, 
            network_definition=[
                [1, 64, 2, True],
                [64, 128, 3, True],
                [128, 256, 4, False],
                [256, 512, 4, False]
            ],conv_filter_size=3,
        ):
        # old style
        super(My_Net, self).__init__()
        #new style
        #super()
        # or
        # super().__init__()

        # collect layers
        self.layers = collections.OrderedDict()
        for i in range(len(network_definition)):
            add_layer_level(self.layers, i, network_definition[i], conv_filter_size=conv_filter_size)
        self.layers['conv{}_{}'.format(5,1) ] = nn.Conv2d(network_definition[-1][1], 128, 1, padding=0)
        self.layers['relu{}_{}'.format(5,2) ] = nn.ReLU(inplace=True)
        self.layers['conv{}_{}'.format(5,3) ] = nn.Conv2d(128, 1, 1, padding=0)

        #self.layers = nn.ModuleList(layers)
        self.network = nn.Sequential(self.layers)
    

    def forward(self, x):
        x = self.network (x)        
        return x

def init_weights(m):
    #print('init_weights:', m)
    #if type(m) == nn.Conv2d:
        #nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        #m.weight.data.xavier_uniform_(1.0)
        #print(m.weight)
    # Frome MobileNet v2
    # "instance" caters for inheritance (an instance of a derived class is an instance 
    # of a base class, too), while checking for equality of type does not (it demands 
    # identity of types and rejects instances of subtypes, AKA subclasses)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.zeros_(m.bias)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    net = My_Net()
    print(net)
    net.apply(init_weights)

    my_root_dir = "/Users/Michelle/Documents/Augsburg_Uni/SS 2019/Bachelorarbeit/Aufnahmen/Aufnahmen_bearbeitet/20190420_Easter_special/Images_Part_1_Adhesion"
    dataset = CellCoreDataset(my_root_dir, 
                            transform = transforms.Compose(
                                [Rescale( int(1864 / 2) ),
                                RandomRescale( 0.9, 1.1 ), 
                                RandomCrop(224)]
                            ),
                            output_reduction=4
                )
    print(len(dataset))

    #for n in range(2,100):
    #    show_training_sample(dataset[n])
    #    plt.pause(1)


    sample = dataset[0]
    img = sample['image']

    x = torch.Tensor([img]).unsqueeze(0)
    out = net(x)
    print(out.shape)
    # print(x)

    out = net(x)
    target = torch.Tensor([sample['gt_map']]).unsqueeze(0)

    # loss function
    # mean-squared error between the input and the target
    #criterion = nn.MSELoss()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=10.0)

    plt.imshow(out.squeeze().detach().numpy())
        #plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.show()  # pause a bit so that plots are updated
    plt.imshow(target.squeeze().numpy())
    plt.show()  # pause a bit so that plots are updated




    # update the weights
    # method: Stochastic Gradient Descent (SGD)
    # formula: weight = weight - learning_rate * gradient

    # create my optimizer with learning rate lr = 0.01
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    train_dataset_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=4, shuffle=True,
                                                num_workers=0)
    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_dataset_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = torch.Tensor(data['image']).unsqueeze(1)
            targets = torch.Tensor(data['gt_map']).unsqueeze(1)

            # plt.imshow(inputs.squeeze().squeeze().detach().numpy())
            # plt.show()  # pause a bit so that plots are updated
            # plt.imshow(targets.squeeze().squeeze().numpy())
            # plt.show()  # pause a bit so that plots are updated

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            # if i % 10 == 9:
            #     plt.imshow(outputs.squeeze().squeeze().detach().numpy())
            #     plt.show()  # pause a bit so that plots are updated
            #     plt.imshow(targets.squeeze().squeeze().numpy())
            #     plt.show()  # pause a bit so that plots are updated

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2 == 1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, 1000 * running_loss / i))
                running_loss = 0.0

    print('Finished Training')
