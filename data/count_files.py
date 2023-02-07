from glob import glob
import json

root_dir = "data/sequence_independent/data_and_cell_center_annotations/"
#root_dir = ""
dirs = ["20190420/", "20190511/", "20190515/", "20190524/"]
#dirs = ["train/", "val/", "test/"]
partition_dirs = ["train", "val", "test"]

print("-" * 60)
print("Per Day")
for direc in dirs:
    files = glob(root_dir + "*/" + direc + "*/*.json")
    print(direc + ": " + str(len(files)))

print("-" * 60)
print("Per Partition")
max_annotations = 0
min_annotations = 100
file_most_annotations = ""
file_least_annotations = ""
for direc in partition_dirs:
    count_annotated_cells = 0
    files = glob(root_dir + direc + "/*/*/*.json")
    for f in files:
        with open(f) as json_file:  
            my_params = json.load(json_file)
            gt_points = my_params['cell_cores']
        n_annotated_cells = len(gt_points)
        if n_annotated_cells > max_annotations:
            max_annotations = n_annotated_cells
            file_most_annotations = f
        if n_annotated_cells < min_annotations:
            file_least_annotations = f
            min_annotations = n_annotated_cells
        count_annotated_cells += n_annotated_cells
    print(direc + " - Files: " + str(len(files)) + ", Annotated Cells: " + str(count_annotated_cells))
print("Most annotations in " + file_most_annotations + ": " + str(max_annotations))
print("Least annotations in " + file_least_annotations + ": " + str(min_annotations))


    
