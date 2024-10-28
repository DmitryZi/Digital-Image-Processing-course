import cv2 as cv
import numpy as np
from fish_distorsion_applier import fish_distorsion_apllier


class param_manager:

    MIN_DIST = -1
    MAX_DIST = 10

    def __init__(self):
        self.effect_w_center = 0
        self.effect_h_center = 0
        self.distortion_power = 0.
        self.image = None

    def increase_distorsion(self):
        ADD = 0.1
        self.__change_dist(ADD)

    def decrease_distorsion(self):
        ADD = -0.1
        self.__change_dist(ADD)

    def clear(self):
        self.effect_w_center = 0
        self.effect_h_center = 0
        self.distortion_power = 0.

    def update_center(self, w, h):
        self.effect_w_center = w
        self.effect_h_center = h

    def __change_dist(self, add):
        self.distortion_power = max(self.MIN_DIST, min (self.MAX_DIST, round(self.distortion_power + add, 1)))

    def same(self, rhs) -> bool:
        same_effect = (self.effect_h_center == rhs.effect_h_center) and (self.effect_w_center == rhs.effect_w_center)
        same_dist = np.isclose(self.distortion_power, rhs.distortion_power)
        return same_effect and same_dist

    def load(self, image_path):
        self.image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        MIN_WIDTH = 800
        if self.image.shape[1] < MIN_WIDTH:
            scale = MIN_WIDTH // self.image.shape[1]
            new_width = self.image.shape[1] * scale
            new_height = self.image.shape[0] * scale

            # resize image
            self.image = cv.resize(self.image, (new_width, new_height), interpolation = cv.INTER_AREA)

        self.image = fish_distorsion_apllier.prepocess(self.image)



class screen:
    STATUS_DONE = "Done"
    STATUS_INPROGRESS = "In progress..."
    STATUS_EMPTY = ""

    CLOSE_STATE = -1
    OFF_STATE = 1
    WORK_STATE = 0
    PASS_STATE = 2

    def __init__(self, window_name):
        self.__manager = param_manager()
        self.__curr_frame = None
        self.status = self.STATUS_EMPTY
        self.__state = self.OFF_STATE
        self.status_bar_text = None
        self.window_name = window_name
        self.update_status_bar()


    def __generate_status_bar_text(self):
        text = f"FishEye. Center={self.__manager.effect_w_center, self.__manager.effect_h_center}, Power={self.__manager.distortion_power}"
        if self.status != self.STATUS_EMPTY:
            text += f" Status: {self.status}"
        return text

    def update_status_bar(self):
        self.status_bar_text = self.__generate_status_bar_text()

    def mouse_click(self, event, x, y):

        if event == cv.EVENT_LBUTTONDOWN:
            self.__manager.update_center(y, x)
            self.status = self.STATUS_EMPTY
            self.__state == self.PASS_STATE
            self.draw_screen()

        if event == cv.EVENT_RBUTTONDOWN:
            self.__state = self.PASS_STATE
            self.status = self.STATUS_EMPTY
            self.__manager.clear()
            self.draw_screen()

    def manage_input(self):
        key = cv.waitKeyEx(0)

        if cv.getWindowProperty(self.window_name, cv.WND_PROP_VISIBLE) < 1:
            self.__state = self.CLOSE_STATE
            return
        if key < 0:
            self.__state = self.CLOSE_STATE
            return

        if key == ord('+') or key == ord('='):
            self.__manager.increase_distorsion()
            self.__state = self.PASS_STATE
            self.status = self.STATUS_EMPTY
        if key == ord('-') or key == ord('_'):
            self.__manager.decrease_distorsion()
            self.__state = self.PASS_STATE
            self.status = self.STATUS_EMPTY
        elif key == ord('q') or key == ord('Q') or key == 27:
            self.__state = self.CLOSE_STATE
        elif key == 13:
            # Enter
            self.status = self.STATUS_INPROGRESS
            self.__state = self.WORK_STATE

    def loop(self, image_path):
        self.__manager.load(image_path)
        self.__curr_frame = self.__manager.image.copy()
        self.draw_screen()
        cv.setMouseCallback(self.window_name, lambda event, x, y, _a , _b: self.mouse_click(event, x, y))

        while self.__state != self.CLOSE_STATE:
            self.manage_input()
            self.draw_screen()
            if self.__state == self.WORK_STATE:
                cv.waitKey(1)
                if not np.isclose(self.__manager.distortion_power, 0.):
                    self.__curr_frame = fish_distorsion_apllier.apply_fish_distortion(self.__manager.image,
                                                                                    self.__manager.effect_w_center,
                                                                                    self.__manager.effect_h_center,
                                                                                    self.__manager.distortion_power)
                else:
                    self.__curr_frame = self.__manager.image.copy()
                self.status = self.STATUS_DONE
                self.__state = self.PASS_STATE
            self.draw_screen()

        cv.destroyAllWindows()

    def get_status_bar(self):
        status_bar = np.zeros((30, self.__curr_frame.shape[1], 4), dtype=np.uint8)
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(status_bar, self.status_bar_text ,(10, 25), font, 0.4, (255,255,255), 1)
        return status_bar

    def draw_screen(self):
        self.update_status_bar()
        # print(self.__curr_frame.shape, self.get_status_bar().shape)
        screen = np.vstack((self.__curr_frame.copy (), self.get_status_bar().copy ()))
        # cv.displayStatusBar(self.window_name, self.status_bar_text)
        cv.imshow(self.window_name, screen)





if __name__ == "__main__":
    test = screen("Fisheye")
    test.loop('test_pic2.jpg')
