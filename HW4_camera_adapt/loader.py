from os import path, listdir
from debug_info import debug_message
import cv2 as cv

def does_file_exist(file_name) -> bool:
    return path.exists(file_name) and path.isfile(file_name)

def does_dir_exist(dir_name):
    return path.exists(dir_name) and path.isdir(dir_name)

def all_files_from_dir(dir_name) -> list:
    if not does_dir_exist(dir_name):
        return []
    return [path.join(dir_name, file_name) for file_name in listdir(dir_name) \
            if does_file_exist(path.join(dir_name, file_name))]


def load_image(file_name):
    if does_file_exist(file_name):
        image = cv.imread(file_name, cv.IMREAD_UNCHANGED)
        return image
    debug_message(f"Can't read image from file {file_name}")
    return None
