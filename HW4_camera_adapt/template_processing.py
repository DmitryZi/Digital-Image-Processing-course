import cv2 as cv
import numpy as np
from loader import load_image
from debug_info import debug_message
from image_processing import get_border_contours
from sudoku_solver import CELLS_COUNT, EMPTY_CELL_VALUE


def board_cut(board_image):

    cell_size = np.array((board_image.shape[0] // CELLS_COUNT, board_image.shape[1] // CELLS_COUNT))
    offset = (cell_size * 0).astype(np.int32)

    result = np.zeros((CELLS_COUNT, CELLS_COUNT, 2, 2), dtype=np.int32)

    for cell_h_ind in range(CELLS_COUNT):
        for cell_w_ind in range(CELLS_COUNT):
            start_x, start_y = np.maximum(cell_size * [cell_h_ind, cell_w_ind] - offset, 0)
            end_x, end_y = np.maximum(cell_size * [cell_h_ind + 1, cell_w_ind + 1] + offset, 0)
            result[cell_h_ind, cell_w_ind] = [[start_x, start_y], [end_x, end_y]]
            # print(result[cell_h_ind, cell_w_ind])

    return result, cell_size

def scale_templates(in_templates, scale):
    result = []
    for template in in_templates:
        new_width = int(template.shape[1] * scale)
        new_height = int(template.shape[0] * scale)
        new_template = cv.resize(template, (new_width, new_height))
        result.append(new_template.astype(template.dtype))
        # cv.imshow('template_after', template)
        # cv.waitKey(0)
    return result

def preprocess_image(in_image):
    if (in_image.ndim > 2):
        res_image = cv.cvtColor(in_image, cv.COLOR_BGR2GRAY)
    else:
        res_image = in_image.copy()
    # res_image = cv.Canny(res_image, 50, 200)
    _, res_image = cv.threshold(res_image, 155, 255, cv.THRESH_BINARY)
    res_image = res_image / 255.
    '''
    contours = get_border_contours(res_image)
    min_cnt = contours[0]

    for x in range(res_image.shape[0]):
        for y in range(res_image.shape[1]):
            res_image[x, y] = 255 if cv.pointPolygonTest(min_cnt, (y, x), False) < 0 else res_image[x, y]
    '''
    return res_image.astype(np.float32)


def predict_template(image, in_templates):
    MIN_MATCH = 0.4
    result = []

    # cv.imshow("Test_before", image)
    test_image = preprocess_image(image)
    # cv.imshow("Test", test_image)
    # cv.waitKey(0)
    for template in in_templates:
        template_res = cv.matchTemplate(test_image, template, cv.TM_CCOEFF_NORMED)
        min_val , max_val, min_loc, max_loc = cv.minMaxLoc(template_res)
        '''
        if 0.1 * image.shape[1] < max_loc[0] < 0.1 * image.shape[1] and \
           0.1 * image.shape[0] < max_loc[1] < 0.1 * image.shape[0]:
            result.append(max_val)
        else:
            result.append(0)
        '''
        result.append(max_val)
        # result.append(1 - min_val)

    # print(result)
    if max(result) < MIN_MATCH:
        return -1

    res = np.argmax(result)
    # cv.namedWindow("test", cv.WINDOW_KEEPRATIO)
    # cv.imshow("test", test_image)
    # cv.namedWindow("match", cv.WINDOW_KEEPRATIO)
    # cv.imshow("match", in_templates[res])
    # cv.waitKey(0)
    return res

def grid_create(board_image, cells_borders, cells_size, num_templates):

    result = np.full((CELLS_COUNT, CELLS_COUNT), EMPTY_CELL_VALUE, dtype=np.int32)

    # Digit is 65% height of cell
    orig_template_size = num_templates[0].shape[0]
    scale = (0.7 * cells_size[0]) / orig_template_size
    scaled_templates = scale_templates(num_templates, scale)

    for cell_h_ind in range(CELLS_COUNT):
        for cell_w_ind in range(CELLS_COUNT):
            cell_start, cell_end = cells_borders[cell_h_ind, cell_w_ind]
            # print(cell_start, cell_end)

            cell_image = board_image[cell_start[0]:cell_end[0], cell_start[1]:cell_end[1]]
            found_template = predict_template(cell_image, scaled_templates)
            if found_template >= 0:
                result[cell_h_ind, cell_w_ind] = found_template + 1
            # cv.imshow(f"h={cell_h_ind} w={cell_w_ind}", cell_image)
            # cv.waitKey(0)
    # print(result)
    return result

def load_templates(templates_paths):

    result = []
    for template_number_path in templates_paths:
        num_image = load_image(template_number_path)
        if num_image is None:
            debug_message(f"Can't load template number image")
            return None
        num_image = preprocess_image(num_image)
        # num_image = cv.cvtColor(num_image, cv.COLOR_BGR2GRAY)
        # _, num_image = cv.threshold(num_image, 200, 255, cv.THRESH_BINARY)
        # cv.imshow("Template", num_image)
        # cv.waitKey(0)
        result.append(num_image)

    return result