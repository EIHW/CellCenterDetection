# -*- coding: utf-8 -*-
 
import os
import json
import cv2


# root directory 
my_root_dir = "/Volumes/NN/Aufnahmen_bearbeitet/test/20190524/Chip03_15dbm_20perc"

# get all files in directories starting from root directory (my_root_dir)
# only get .tif files 
# returns list of filenames
def get_files(my_root_dir, ext_list=['.tif']):
    my_file_list = []
    for (dirpath, dirnames, filenames) in os.walk(my_root_dir):
        for filename in filenames:
            basename, ext = os.path.splitext(filename)
            if ext in ext_list:
                my_file_list.append(os.path.join(dirpath, filename))
            print(os.path.join(dirpath, filename))
    return my_file_list


# mouse callback function
# annotation of ground truth
def my_mousehandler_for_gt_annotations(event,x,y,flags,param):
    
    # mouse LEFT click:
    # get shortcuts and scaling factor (usually scale_factor = 2)
    # add clicked cell core to cell_cores list
    cell_cores = my_params['cell_cores']
    if event == cv2.EVENT_LBUTTONDOWN:
        scale_factor = my_params['scaling']
        cell_cores.append([x*scale_factor,y*scale_factor])
    
    # mouse RIGHT click:
    # delete clicked cell core out of cell_cores list
    # delete the closest cell core to right click cell
    if event == cv2.EVENT_RBUTTONDOWN:
        x=x * cur_scaling
        y=y * cur_scaling
        min_dist = 10000000.0
        min_index = -1
        for i in range(len(cell_cores)):
            x2,y2 = cell_cores[i]
            #print(cell_cores[i], x,y)
            dist = (x2-x)*(x2-x)+(y2-y)*(y2-y)
            if dist < min_dist:
                min_dist = dist
                min_index = i
        if min_index != -1:
            del(cell_cores[min_index])
            #print('Lösche ', min_index, cell_cores[min_index])


# draw annotated cell cores in according image (RED circle)
# cur_scaling = current scaling factor
# returns image with drawn in cell cores in RED
def draw_annotations(src_img, my_params, cur_scaling):
    fx = 1.0 / cur_scaling
    fy = 1.0 / cur_scaling
    # dst_img = src_img.copy()
    dst_img = cv2.resize(src_img, None, fx=fx, fy=fy)
    cell_cores = my_params['cell_cores']
    for i in range(len(cell_cores)):
        x,y = cell_cores[i]
        x = int(x * fx)
        y = int(y * fy)
        # draw RED circle in image
        cv2.circle(dst_img,(x,y),2,(0,0,255),-1)
    return dst_img


# file can be used as standalone program or reusable module
# the following isn't executable for reuse
# switching between pictures with key codes:
# <-    one image backward  key code: 2 or 106
# ->    one image forward   key code: 3 or 107
# ^     10 images backward  key code: 0 or 105
# v     10 images forward   key code: 1 or 109
# esc   exit 
if __name__ == "__main__":
    # get files sorted
    my_file_list = sorted(get_files(my_root_dir))

    key_code = 0
    index = 0
    old_index = -1
    my_params = {}
    cur_scaling = 2
    my_params['scaling'] = cur_scaling # == 1.0 / 2

    cv2.namedWindow("Cells", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Cells', my_mousehandler_for_gt_annotations, my_params)

    while key_code != 27: # key code 27: esc

        filename = my_file_list[index]

        if old_index != index:
            # load image plus annotation data
            img = cv2.imread(filename) #, cv2.IMREAD_GRAYSCALE)

            basename, ext = os.path.splitext(filename)
            filename_anno = basename + '.json'
            my_params['cell_cores'] = []
            if os.path.exists(filename_anno):
                with open(filename_anno) as json_file:  
                    my_params = json.load(json_file)
            my_params['scaling'] = cur_scaling

            old_index = index
            print('Current image', index, filename)

        dst = draw_annotations(img, my_params, cur_scaling)
        cv2.imshow("Cells",dst)

        key_code = cv2.waitKey(1)
        if key_code > 0:
            print (key_code, index)

        if key_code == 2 or key_code == 106 : # <-     one image backward
            index = max(0, index - 1)
        elif key_code == 0 or key_code == 105: # ^    10 images backward
                index = max(0, index - 10)
        elif key_code == 3 or key_code == 107: # ->   one image forward
            index = min(len(my_file_list) - 1, index + 1)
        elif key_code == 1 or key_code == 109: # v    10 images forward
            index = min(len(my_file_list) - 1, index + 10)
        elif key_code == 115: # write annotation file on 's'
            print('SAVE ANNOTATIONS:', filename_anno)
            with open(filename_anno, 'w') as outfile:  
                json.dump(my_params, outfile)
        else:
            pass

