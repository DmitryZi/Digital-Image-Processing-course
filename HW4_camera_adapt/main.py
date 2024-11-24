import cv2 as cv
from loader import load_image
import image_processing

if __name__ == "__main__":
    TEST_PATH = "examples/6.jpg"
    orig_image = load_image(TEST_PATH)
    if orig_image is not None:
        cv.imshow('Orig', orig_image)
        cv.waitKey(0)

        prepared = image_processing.image_preprocess(orig_image)
        cv.imshow('After', prepared)
        cv.waitKey(0)

        sudoku_cnts = image_processing.get_border_contours(prepared)
        if sudoku_cnts is not None:
            res_contours = cv.drawContours(orig_image, sudoku_cnts, -1, (0, 255, 0), 2)
            cv.imshow('Contour', res_contours)
            cv.waitKey(0)

            rect_approx = image_processing.approx_as_rect(sudoku_cnts)
            res_contours = cv.drawContours(orig_image, [rect_approx], -1, (0, 0, 255), 2)
            cv.imshow('Contour approx', res_contours)
            cv.waitKey(0)
