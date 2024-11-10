from os import path, listdir
from zipfile import ZipFile
from debug_info import debug_message

def does_file_exist(file_name) -> bool:
    return path.exists(file_name) and path.isfile(file_name)

def does_dir_exist(dir_name):
    return path.exists(dir_name) and path.isdir(dir_name)

def all_files_from_dir(dir_name) -> list:
    if not does_dir_exist(dir_name):
        return []
    return [path.join(dir_name, file_name) for file_name in listdir(dir_name) \
            if does_file_exist(path.join(dir_name, file_name))]

def all_files_with_ext(dir_name, ext) -> list:
    return [file_name for file_name in all_files_from_dir(dir_name) if ext == slpit_by_name_and_ext(file_name)[1]]

def slpit_by_name_and_ext(file_path):
    return path.splitext(file_path)

def unzip_to_dir(file, dir_path):
    if not does_file_exist(file):
        return

    with ZipFile(file, 'r') as zip_file:
        zip_file.extractall(dir_path)

def extract_all_data(zip_file_name):
    dir_path, _ = slpit_by_name_and_ext(zip_file_name)

    if not does_dir_exist(dir_path):
        # Not extracted yet
        unzip_to_dir(zip_file_name, dir_path)

    if not does_dir_exist(dir_path):
        debug_message(f"Can't load data from zip {zip_file_name} and open {dir_path}")
        return False

    return True

def load_test_res_files(zip_file_name, train_dir, test_dir, files_ext = '.png'):

    if not extract_all_data(zip_file_name):
        return [], []

    data_dir, _ = slpit_by_name_and_ext(zip_file_name)
    test_dir_abs_path = path.join(data_dir, test_dir)
    train_dir_abs_path = path.join(data_dir, train_dir)
    if not does_dir_exist(test_dir_abs_path):
        debug_message(f"Can't find {test_dir} (abs path {test_dir_abs_path}) \
                      in unzipped {data_dir}")
        return [], []
    if not does_dir_exist(train_dir_abs_path):
        debug_message(f"Can't find {train_dir} (abs path {train_dir_abs_path}) \
                      in unzipped {data_dir}")
        return [], []

    return all_files_with_ext(train_dir_abs_path, files_ext), \
           all_files_with_ext(test_dir_abs_path, files_ext)



if __name__ == "__main__":
    ZIP_FILE_NAME = "archive.zip"
    TRAIN_DIR = path.join("all_images", "images")
    TEST_DIR = path.join("all_masks", "masks")
    print(load_test_res_files(ZIP_FILE_NAME, TRAIN_DIR, TEST_DIR))
