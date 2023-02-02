from glob import glob

root_dir = "/nas/staff/data_work/manuel/cloned_repos/cell_center_detection/CellCenterDetection/data/Michelle/"
#root_dir = ""
dirs = ["20190420/", "20190511/", "20190515/", "20190524/"]
#dirs = ["train/", "val/", "test/"]

for direc in dirs:
    print(root_dir + "*/" + direc + "*/*.json")
    files = glob(root_dir + "*/" + direc + "*/*.json")
    #if direc == 'val/':
    #    for file in files:
    #        print(file)
    print(direc + ": " + str(len(files)))
