from glob import glob
import random
import shutil
import os

root_dir = "sequence_independent/"
target_dir = "random/"
dirs = ["train/", "val/", "test/"]
percentages = {"train/":0.60, "val/":0.2, "test/":0.2}

files_wildcard = root_dir + "*/*/*/*.json"
print(files_wildcard)
files = glob(files_wildcard)
current_perc = 0.
random.shuffle(files)
print(len(files))
for partition in dirs:
    files_for_partition = files[int(current_perc * len(files)):int((current_perc + percentages[partition]) * len(files))]
    current_perc += percentages[partition]
    print("-----------------------------------------")
    print(partition + " Len: " + str(len(files_for_partition)))
    print("-----------------------------------------")
    for json_file in files_for_partition:
        json_dest_file = json_file.replace(root_dir, target_dir)
        img_file = json_file.replace(".json", ".tif")
        if os.path.isfile(img_file):
            img_dest_file = img_file.replace(root_dir, target_dir)
            for partition_another_index in dirs:
                img_dest_file = img_dest_file.replace(partition_another_index, partition)
                json_dest_file = json_dest_file.replace(partition_another_index, partition)
            os.makedirs(os.path.dirname(img_dest_file), exist_ok=True)
            shutil.copy(json_file, json_dest_file)
            shutil.copy(img_file, img_dest_file)
        else:
            print("File doesn't exist: " + img_file)


