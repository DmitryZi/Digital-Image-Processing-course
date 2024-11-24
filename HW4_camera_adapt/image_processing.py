import cv2 as cv
from debug_info import debug_message

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