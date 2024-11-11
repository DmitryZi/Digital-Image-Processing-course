from loader import does_file_exist
from debug_info import debug_message
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def load_images_from_files(file_list: list):
    result = []

    for file_name in file_list:
        if does_file_exist(file_name):
            image = cv.imread(file_name, cv.IMREAD_UNCHANGED)
            if image.ndim > 2:
                # correct RGB
                image = image[:, :, :: -1]
            result.append(image)
        else:
            debug_message(f"File {file_name} is not found and will be skipped, no image")

    return np.array(result)

def draw_plt_image(image, row_range = None, col_range = None):
    rows, cols = image.shape[:2]

    draw_row_range = row_range if row_range is not None else (0, rows)
    draw_col_range = col_range if col_range is not None else (0, cols)

    plt.imshow(image[draw_row_range[0]:draw_row_range[1], draw_col_range[0]:draw_col_range[1]], interpolation = 'none' )

def process_images(image_list, function):
    result = []
    for image in image_list:
        result.append(function(image))

    return np.array(result)
