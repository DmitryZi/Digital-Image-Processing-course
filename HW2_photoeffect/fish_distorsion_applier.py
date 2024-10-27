import numpy as np
from math import sqrt
from debug_info import debug_message
import cv2 as cv
from multiprocessing import Process, Manager


class fish_distorsion_apllier:

    @staticmethod
    def prepocess(in_image):
        """
        Преобразование входного изображения к RGBA формату
        In
        ----
        in_image: numpy.ndarray, BW, RGB or RGBA image
        Out
        ----
        out_image: numpy.ndarray, RGBA image
        """
        IMAGE_DTYPE = np.uint8
        RGB_CHANNELS = 3
        RGBA_CHANNELS = 4
        A_CHANNEL_VAL = 255

        if in_image.ndim < 2:
            debug_message(f"Unable to preprocess image, not enough dims, {2} requied (got {in_image.ndim})")
            return None
        if in_image.dtype != IMAGE_DTYPE:
            debug_message(f"Unable to preprocess image, dtype is not {IMAGE_DTYPE} (got {in_image.dtype})")
            return None

        im_width = in_image.shape[0]
        im_height = in_image.shape[1]
        out_image = np.zeros((im_width, im_height, 4), dtype=np.uint8)

        if in_image.ndim == 2:
            # copy BW image to RGB
            for channel_num in range(RGB_CHANNELS):
                out_image[..., channel_num] = in_image.copy()
            # alpha channel
            out_image[..., RGBA_CHANNELS - 1] = A_CHANNEL_VAL
        elif in_image.ndim == 3:
            channels_count = in_image.shape[2]
            if channels_count == RGB_CHANNELS:
                # copy RGB image to RGB
                out_image[..., :RGB_CHANNELS] = in_image.copy()
                # alpha channel
                out_image[..., RGBA_CHANNELS - 1] = A_CHANNEL_VAL
            else:
                # copy RGBA image
                out_image = in_image.copy()
        else:
            debug_message(f"Unable to preprocess image, uknown dims count, {2} or {3} requied (got {in_image.ndim})")
            return None

        return out_image

    @staticmethod
    def get_fish_xn_yn(dst_w, dst_h, distortion):
        """
        Для к-т точки в новом изображении строится их прообраз в исходном изображении
        In
        ----
        dst_w: in [-1, 1]
        dst_h: in [-1, 1]
        distortion: value of image distortion, bigger value => pixels move from effect center
        Out
        ----
        src_w: in [-1, 1], x pixel pos in source image
        src_h: in [-1, 1], y pixel pos in source image
        success: bool, success of transform
        """

        dst_rad = sqrt(dst_w**2 + dst_h**2)
        norm_coef = min(1, max(0, 1 - distortion * (dst_rad**2)))
        if np.isclose(norm_coef, 0.):
            return None, None, False

        return dst_w / norm_coef, dst_h / norm_coef, True


    @staticmethod
    def apply_fish_distortion(in_image, effect_w_center, effect_h_center, distortion_coefficient, use_parallel = True):
        """
        Применяет эффект fisheye в указанном центре изображения
        In
        ----
        in_image: numpy.ndarray, RGBA image
        effect_w_center, effect_h_center: effect center in pixels
        distortion_coefficient: value of image distortion, bigger value => pixels move from effect center
        use_parallel: bool, use parallel calculations
        Out
        ----
        out_image: numpy.ndarray, RGBA image with effect
        """

        if not fish_distorsion_apllier.is_valid_image(in_image):
            debug_message('Unable to apply fish distortion to image, preprocess it first')
            return None

        if use_parallel:
            MAX_TOTAL_LAUNCHES = 20
            return fish_distorsion_apllier.__apply_fish_distortion_mp(in_image, MAX_TOTAL_LAUNCHES, effect_w_center, effect_h_center, distortion_coefficient)
        else:
            return fish_distorsion_apllier.__apply_fish_distortion_mp(in_image, 1, effect_w_center, effect_h_center, distortion_coefficient)


    @staticmethod
    def __apply_fish_distortion_mp(in_image, process_count, effect_w_center, effect_h_center, distortion_coefficient):
        """
        Применяет эффект fisheye в указанном центре изображения
        In
        ----
        in_image: numpy.ndarray, RGBA image
        process_count: int, amount of total parallel calls
        effect_w_center, effect_h_center: effect center in pixels
        distortion_coefficient: value of image distortion, bigger value => pixels move from effect center
        Out
        ----
        out_image: numpy.ndarray, RGBA image with effect
        """

        im_width = in_image.shape[0]
        w_all_range = np.arange(im_width)
        out_image = np.zeros_like(in_image)
        w_process_ranges = np.array_split(w_all_range, process_count)
        in_args = [[in_image, w_range, effect_w_center, effect_h_center, distortion_coefficient] for w_range in w_process_ranges]
        return_dict = fish_distorsion_apllier.__parallel_launch(fish_distorsion_apllier.process_apply_fish_distortion, in_args)

        for width_range, image_res in return_dict.items():
            # print(widht_range, image_res.shape)
            out_image[width_range[0]:width_range[1]] = image_res
        return out_image


    @staticmethod
    def __parallel_launch(launch_target, args_list):
        MAX_PARALLEL_LAUNCHES = 20
        proc_list = []
        manager = Manager()
        return_dict = manager.dict()
        launch_chunks_count = (len(args_list) + MAX_PARALLEL_LAUNCHES - 1) // MAX_PARALLEL_LAUNCHES
        for chunk_num in range(launch_chunks_count):
            launch_count = min(MAX_PARALLEL_LAUNCHES, len(args_list) - chunk_num * MAX_PARALLEL_LAUNCHES)
            proc_list.clear()
            for task_num in range(launch_count):
                proc = Process(target=launch_target, args=[return_dict] + args_list[chunk_num * MAX_PARALLEL_LAUNCHES + task_num])
                proc.start()
                proc_list.append(proc)

            for process in proc_list:
                process.join()

        return return_dict

    @staticmethod
    def process_apply_fish_distortion(out_image_dict, in_image, w_range, effect_w_center, effect_h_center, distortion_coefficient):
        """
        Применяет эффект fisheye в указанном центре изображения
        In
        ----
        in_image: numpy.ndarray, RGBA image
        out_image_dict: dict, key is w_range, value is numpy.ndarray, RGBA image with effect
        w_range: range/list, pixel_width_values
        effect_w_center, effect_h_center: effect center in pixels
        distortion_coefficient: value of image distortion, bigger value => pixels move from effect center
        Out
        ----
        out_image: numpy.ndarray, RGBA image with effect
        """


        im_width = in_image.shape[0]
        im_height = in_image.shape[1]
        out_image = np.zeros((len(w_range), im_height, 4), dtype=np.uint8)

        EMPTY = np.zeros(4, dtype=np.uint8)

        for w_ind in w_range:
            for h_ind in range(im_height):

                out_w_norm, out_h_norm = fish_distorsion_apllier.normalize_pixel_pos(w_ind, h_ind,
                                                                                       effect_w_center, effect_h_center,
                                                                                       im_width, im_height)

                src_w_norm, src_h_norm, is_ok = fish_distorsion_apllier.get_fish_xn_yn(out_w_norm, out_h_norm, distortion_coefficient)

                dst_pixel_value = EMPTY
                if is_ok:
                    src_w_ind, src_h_ind = fish_distorsion_apllier.unormalize_pixel_pos(src_w_norm, src_h_norm,
                                                                                          effect_w_center, effect_h_center,
                                                                                          im_width, im_height)
                    if src_w_ind is not None and src_h_ind is not None:
                        dst_pixel_value = in_image[src_w_ind, src_h_ind]

                out_image[w_ind - w_range[0], h_ind] = dst_pixel_value.copy()

        out_image_dict[(w_range[0], w_range[-1] + 1)] = out_image.copy()

    @staticmethod
    def __get_max_possible_size(effect_w_center, effect_h_center, im_width, im_height):
        """
        Получить максимальное возможное отклонение от центра эффекта для заданного изображения
        In
        ---
        effect_w_center, effect_h_center: int, effect center, future (0, 0)
        im_width, im_height: int, full image size
        Out
        ----
        max_width, max_height: int, max possible distance from effect center
        """
        max_width = max(abs(im_width - effect_w_center), abs(effect_w_center))
        max_height = max(abs(im_height - effect_h_center), abs(effect_h_center))
        return max_width, max_height


    @staticmethod
    def normalize_pixel_pos(w_ind, h_ind, effect_w_center, effect_h_center, im_width, im_height):
        """
        Приводит положение пикселя в диапазон [-1, 1], где (0, 0) - центр эффекта
        In
        ----
        w_ind, h_ind: int, pixel pos
        effect_w_center, effect_h_center: int, effect center, future (0, 0)
        im_width, im_height: int, full image size
        Out
        ---
        norm_width, nor_height: int, pixel pos in range [-1, 1]
        """
        max_possible_width, max_possible_height = fish_distorsion_apllier.__get_max_possible_size(effect_w_center, effect_h_center,
                                                                                                  im_width, im_height)
        norm_width = float(w_ind - effect_w_center) / float(max_possible_width)
        norm_height = float(h_ind - effect_h_center) / float(max_possible_height)
        return norm_width, norm_height


    @staticmethod
    def unormalize_pixel_pos(norm_w, norm_h, effect_w_center, effect_h_center, im_width, im_height):
        """
        Обратное к normalize_pixel_pos
        In
        ----
        norm_w, norm_h: float, pixel pos in [-1, 1]
        effect_w_center, effect_h_center: int, effect center, future (0, 0)
        im_width, im_height: int, full image size
        Out
        ---
        w_ind, h_ind: int, pixel pos
        """
        max_possible_width, max_possible_height = fish_distorsion_apllier.__get_max_possible_size(effect_w_center, effect_h_center,
                                                                                            im_width, im_height)
        res_width = round(norm_w * float(max_possible_width) + effect_w_center)
        res_height = round(norm_h * float(max_possible_height) + effect_h_center)

        in_bounds = lambda val, max_val: min(max(val, 0), max_val - 1) == val
        if not in_bounds(res_width, im_width) or not in_bounds(res_height, im_height) or not(-1. <= norm_w <= 1) or not(-1. <= norm_h <= 1):
            return None, None

        return res_width, res_height

    @staticmethod
    def is_valid_image(in_image) -> bool:
        """
        Проверяет корректность входного изображения перед обработкой
        In
        ----
        in_image: numpy.ndarray
        Out
        ----
        is_valid: bool, корректность изображения
        """
        REQUIRED_DIMS = 3
        REQUIRED_CHANNELS = 4
        if in_image.ndim != REQUIRED_DIMS:
            return False
        if in_image.shape[REQUIRED_DIMS - 1] != REQUIRED_CHANNELS:
            return False
        return True



def preprocess_test():
    test = 'test_pic.jpg'
    orig = cv.imread(test, cv.IMREAD_UNCHANGED)
    print(orig.shape)
    cv.imshow('Orig', orig)
    cv.waitKey(0)
    rgba = fish_distorsion_apllier.prepocess(orig)
    print(rgba.shape)
    cv.imshow('RGBA', rgba)
    cv.waitKey(0)


def transform_test():
    SIZE = 100
    for w in range(SIZE):
        for h in range(SIZE):
            out_w_norm, out_h_norm = fish_distorsion_apllier.normalize_pixel_pos(w, h,
                                                                                   0, 0,
                                                                                   SIZE, SIZE)
            src_w, src_h = fish_distorsion_apllier.unormalize_pixel_pos(out_w_norm, out_h_norm,
                                                                          0, 0,
                                                                          SIZE, SIZE)
            if w != src_w or h != src_h:
                debug_message(f"WIDTH {w} != {src_w}, HEIGHT {h} != {src_h}")


def no_dist_test():
    test = 'test_pic.jpg'
    orig = cv.imread(test, cv.IMREAD_UNCHANGED)
    cv.imshow('Orig', orig)
    cv.waitKey(0)
    rgba = fish_distorsion_apllier.prepocess(orig)
    res = fish_distorsion_apllier.apply_fish_distortion(rgba,
                                                        200, 200, 0.)
    cv.imshow('NO_DIST', res)
    cv.waitKey(0)

def center_dist_test():
    test = 'test_pic.jpg'
    orig = cv.imread(test, cv.IMREAD_UNCHANGED)
    cv.imshow('Orig', orig)
    cv.waitKey(0)
    rgba = fish_distorsion_apllier.prepocess(orig)
    res = fish_distorsion_apllier.apply_fish_distortion(rgba,
                                                        rgba.shape[0] // 2, rgba.shape[1] // 2,
                                                        2)
    cv.imshow('CENTER_DIST', res)
    cv.waitKey(0)
    res = fish_distorsion_apllier.apply_fish_distortion(rgba,
                                                        rgba.shape[0] // 2, rgba.shape[1] // 2,
                                                        -2)
    cv.imshow('CENTER_DIST', res)
    cv.waitKey(0)

def edge_dist_test():
    test = 'test_pic.jpg'
    orig = cv.imread(test, cv.IMREAD_UNCHANGED)
    cv.imshow('Orig', orig)
    cv.waitKey(0)
    rgba = fish_distorsion_apllier.prepocess(orig)
    res = fish_distorsion_apllier.apply_fish_distortion(rgba,
                                                        rgba.shape[0] // 4, rgba.shape[1] // 4,
                                                        0.6)
    cv.imshow('CENTER_DIST', res)
    cv.waitKey(0)
    res = fish_distorsion_apllier.apply_fish_distortion(rgba,
                                                        rgba.shape[0] // 4, rgba.shape[1] // 4,
                                                        -0.6)
    cv.imshow('CENTER_DIST', res)
    cv.waitKey(0)

def out_of_bound_dist_test():
    test = 'test_pic.jpg'
    orig = cv.imread(test, cv.IMREAD_UNCHANGED)
    cv.imshow('Orig', orig)
    cv.waitKey(0)
    rgba = fish_distorsion_apllier.prepocess(orig)
    res = fish_distorsion_apllier.apply_fish_distortion(rgba,
                                                        int(rgba.shape[0] * 1.1), int(rgba.shape[1] * 1.1),
                                                        0.6)
    cv.imshow('CENTER_DIST', res)
    cv.waitKey(0)
    res = fish_distorsion_apllier.apply_fish_distortion(rgba,
                                                        int(rgba.shape[0] * 1.1), int(rgba.shape[1] * 1.1),
                                                        -0.6)
    cv.imshow('CENTER_DIST', res)
    cv.waitKey(0)


if __name__ == "__main__":
    #transform_test ()
    #no_dist_test()
    center_dist_test()
    #edge_dist_test()
    #out_of_bound_dist_test()
    pass
