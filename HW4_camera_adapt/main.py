import cv2 as cv
from loader import load_image
import image_processing
import template_processing
from sudoku_solver import solve


if __name__ == "__main__":

    TEST_PATH = "examples/3.jpg"
    NUMBER_PATHS = [f"number_templates/{num}_new.png" for num in range(1, 10)]

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
        # cv.imshow('Contour approx', res_contours)
        # cv.waitKey(0)

        ordered_corners = image_processing.sort_corners(rect_approx)
        transformed, calibrate_matrix = image_processing.camera_calibrate(orig_image, ordered_corners)
        cv.imshow('Transformed', transformed)
        cv.waitKey(0)

        board_cut, cell_size = template_processing.board_cut(transformed)
        in_grid = template_processing.grid_create(transformed, board_cut, cell_size, num_templates)
        print("Before solve:")
        print(in_grid)
        solved_grid = solve(in_grid)
        if solved_grid is None:
            quit()
        print("After solve:")
        print(solved_grid)

        printed_grid = image_processing.print_grid(transformed, board_cut, cell_size, in_grid, solved_grid)

        updated = image_processing.camera_inverse_transform(orig_image, printed_grid, ordered_corners, calibrate_matrix)
        cv.imshow('Result', updated)
        cv.waitKey(0)
