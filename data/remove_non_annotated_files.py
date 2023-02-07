from glob import glob
import os


def get_files_without_annotations(my_root_dir, ext_list=['.tif']):
    my_file_list = []
    for (dirpath, dirnames, filenames) in os.walk(my_root_dir):
        for filename in filenames:
            basename, ext = os.path.splitext(filename)
            if ext.lower() in ext_list:
                # check for associated .json file
                filename_anno = basename + '.json'
                if not os.path.exists(os.path.join(dirpath, filename_anno)):
                    my_file_list.append(os.path.join(dirpath, filename))
                print(os.path.join(dirpath, filename))
    return my_file_list

folder = "data/sequence_independent/data_and_cell_center_annotations/"
non_annotated_files = get_files_without_annotations(folder)
for file_tbr in non_annotated_files:
    os.remove(file_tbr)