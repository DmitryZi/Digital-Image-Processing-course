import cv2 as cv
import numpy as np
from debug_info import debug_message
from sudoku_solver import CELLS_COUNT, EMPTY_CELL_VALUE

def image_preprocess(in_image):
    res_image = None

    if in_image.ndim == 3:
        #BGR to gray
        res_image = cv.cvtColor(in_image, cv.COLOR_BGR2GRAY)
    else:
        # Ready image
        res_image = in_image.copy()

    image_y_mult = in_image.shape[0] / 1000
    image_x_mult = in_image.shape[1] / 1000

    res_image = cv.GaussianBlur(res_image, (5, 5), 0)
    res_image = cv.adaptiveThreshold(res_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, int(min(image_x_mult, image_y_mult) * 41) // 2 * 2 + 1, 7)

    struct_square = cv.getStructuringElement(cv.MORPH_RECT, (int(image_x_mult * 4), int(image_y_mult * 4)))
    res_image = cv.dilate(res_image, struct_square, iterations=1)

    struct_rects = [cv.getStructuringElement(cv.MORPH_RECT, (2, int(image_y_mult * 4))), cv.getStructuringElement(cv.MORPH_RECT, (int(image_x_mult * 4), 2))]
    for rect in struct_rects:
        res_image = cv.erode(res_image, rect, iterations=2)

    # res_image = cv.Canny(res_image, 75, 200)
    return res_image


def get_border_contours(in_image):
    cnts_all = cv.findContours(in_image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # Position of contour is different in different opencv versions
    cnts = cnts_all[0] if len(cnts_all) == 2 else cnts_all[1]
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)

    if len(cnts) < 1:
        debug_message("Can't find contours in image")
        return None

    # skip too big contours
    res_cnts = []
    for cnt in cnts:
        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

        if (rightmost[0] - leftmost[0]) / in_image.shape[1] < 0.99 and \
           (topmost[1] - bottommost[1]) / in_image.shape[0] < 0.99:
           res_cnts.append(cnt)
    return res_cnts


def approx_as_rect(in_contours):
    result = []

    for cont in in_contours:
        cont_perimeter = cv.arcLength(cont, True)
        poly_approx = cv.approxPolyDP(cont, 0.01 * cont_perimeter, True)

        points_count = len(poly_approx)
        if points_count == 4:
            return poly_approx

        if 3 <= points_count <= 5:
            return cv.boxPoints(cv.minAreaRect(poly_approx))

    return result

def print_grid(in_board_image, cells_borders, cell_size, in_grid, solved_grid):

    result = in_board_image.copy()
    for cell_h_ind in range(CELLS_COUNT):
        for cell_w_ind in range(CELLS_COUNT):
            if in_grid[cell_h_ind, cell_w_ind] != EMPTY_CELL_VALUE:
                continue

            # Print number on image
            cell_start, cell_end = cells_borders[cell_w_ind, cell_h_ind]
            offset = cell_size * 0.2
            print_start = (int(cell_start[0] + offset[0]), int(cell_end[1] - offset[1]))
            FONT = cv.FONT_HERSHEY_COMPLEX
            font_size = 2 * cell_size[0] / 80
            COLOR = (10, 10, 10)
            cv.putText(result, str(int(solved_grid[cell_h_ind, cell_w_ind])), print_start, FONT, font_size, COLOR, 2)

    return result



def sort_corners(rect_contour):

    result = np.zeros((4, 2), dtype=np.float32)
    rect_contour = np.squeeze(rect_contour)
    coords_sum = np.sum(rect_contour, axis=1)
    coords_diff = np.diff(rect_contour, axis=1)

    """
    0 1
    3 2
    """
    result[0] = rect_contour[np.argmin(coords_sum)]
    result[1] = rect_contour[np.argmin(coords_diff)]
    result[2] = rect_contour[np.argmax(coords_sum)]
    result[3] = rect_contour[np.argmax(coords_diff)]

    return result


def camera_calibrate(in_image, border_corners):

    border_size = cv.contourArea(border_corners)
    calibrated_image_size = int(np.round(np.sqrt(border_size) // 9, -1) * 9)
    # print(calibrated_image_size)

    new_corners = np.array([[0, 0], [calibrated_image_size - 1, 0], [calibrated_image_size - 1, calibrated_image_size - 1], [0, calibrated_image_size - 1]], border_corners.dtype)
    calibrate_matrix = cv.getPerspectiveTransform(border_corners, new_corners)
    result = cv.warpPerspective(in_image, calibrate_matrix, (calibrated_image_size, calibrated_image_size))

    return result, calibrate_matrix


def camera_inverse_transform(orig_image, changed_part, part_contour, calibrate_matrix):

    result = orig_image.copy()
    result_shape = orig_image.shape[:2]

    inverse_calibrate_matrix = np.linalg.inv(calibrate_matrix)
    inversed_change = cv.warpPerspective(changed_part, calibrate_matrix, result_shape[::-1], flags=cv.WARP_INVERSE_MAP)

    # Update only changed part of image
    contour_mask = np.full(result_shape, False)
    for x in range(result_shape[0]):
        for y in range(result_shape[1]):
            contour_mask[x, y] = cv.pointPolygonTest(part_contour, (y, x), False) >= 0

    result[contour_mask] = inversed_change[contour_mask]
    return result
