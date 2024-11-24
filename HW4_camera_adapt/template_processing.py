import cv2 as cv
import numpy as np


def board_cut(board_image):
    CELLS_COUNT = 9
    cell_size = np.array((board_image.shape[0] // CELLS_COUNT, board_image.shape[1] // CELLS_COUNT))
    offset = (cell_size * 0.2).astype(np.int32)

    result = np.zeros((CELLS_COUNT, CELLS_COUNT, 2, 2), dtype=np.int32)

    for cell_w_ind in range(CELLS_COUNT):
        for cell_h_ind in range(CELLS_COUNT):
            start_x, start_y = np.multiply(cell_size, [cell_w_ind, cell_h_ind]) - offset
            end_x, end_y = np.multiply( cell_size, [cell_w_ind + 1, cell_h_ind + 1]) + offset
            result[cell_w_ind, cell_h_ind] = [[start_x, start_x], [end_x, end_y]]
            print(result[cell_w_ind, cell_h_ind])

    return result
