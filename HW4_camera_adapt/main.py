import cv2 as cv
from loader import load_image
import image_processing
import template_processing


if __name__ == "__main__":

    NUMBER_PATHS = [f"number_templates/{num}.jpg" for num in range(1, 10)]
    TEST_PATH = "examples/3.jpg"

    num_templates = template_processing.load_templates(NUMBER_PATHS)
    if num_templates is None:
        quit()

    orig_image = load_image(TEST_PATH)
    if orig_image is None:
        quit()

    cv.imshow('Orig', orig_image)
    cv.waitKey(0)

    prepared = image_processing.image_preprocess(orig_image)
    cv.imshow('After', prepared)
    cv.waitKey(0)

    sudoku_cnts = image_processing.get_border_contours(prepared)
    if sudoku_cnts is not None:
        res_contours = cv.drawContours(orig_image.copy(), sudoku_cnts, -1, (0, 255, 0), 2)
        # cv.imshow('Contour', res_contours)
        # cv.waitKey(0)

        rect_approx = image_processing.approx_as_rect(sudoku_cnts)
        res_contours = cv.drawContours(orig_image.copy(), [rect_approx], -1, (0, 0, 255), 2)
        cv.imshow('Contour approx', res_contours)
        cv.waitKey(0)

        ordered_corners = image_processing.sort_corners(rect_approx)
        transformed, calibrate_matrix = image_processing.camera_calibrate(orig_image, ordered_corners)
        cv.imshow('Transformed', transformed)
        cv.waitKey(0)

        board_cut, cell_size = template_processing.board_cut(transformed)
        template_processing.grid_create(transformed, board_cut, cell_size, num_templates)
        cv.rectangle(transformed, (100, 100), (300, 300), (0, 255, 0), 10)

        updated = image_processing.camera_inverse_transform(orig_image, transformed, ordered_corners, calibrate_matrix)
        cv.imshow('Result', updated)
        cv.waitKey(0)
