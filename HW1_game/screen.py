from game_manager import  GameManager
from dance_floor_data import DanceFloor
from player_move import PlayerMove
from copy import deepcopy
import cv2 as cv
import numpy as np

class FloorDrawer:
    SQUARE_SIZE = 100


    def __init__(self, hor_pos: int, ver_pos: int):
        self.__corner_hor = hor_pos
        self.__corner_ver = ver_pos

    def draw(self, background, floor: DanceFloor) -> None:
        for row in range(floor.get_height()):
            for column in range(floor.get_width()):
                self.__draw_cell(background, floor.get_cell_color(row, column), row, column)
        self.__draw_grid(background, floor.get_height(), floor.get_width())

    def __draw_grid(self, background, rows: int, columns: int):
        BORDER_THICKNESS = 2
        BORDER_COLOR = (0, 0, 0)  # Black
        for row in range(rows + 1):
            cv.line(background,
                    (self.__corner_hor + row * self.SQUARE_SIZE, self.__corner_ver),
                    (self.__corner_hor + row * self.SQUARE_SIZE, self.__corner_ver + columns * self.SQUARE_SIZE),
                    BORDER_COLOR, BORDER_THICKNESS)
        for column in range(columns + 1):
            cv.line(background,
                    (self.__corner_hor, self.__corner_ver + column * self.SQUARE_SIZE),
                    (self.__corner_hor + rows * self.SQUARE_SIZE, self.__corner_ver + column * self.SQUARE_SIZE),
                    BORDER_COLOR, BORDER_THICKNESS)

    def __draw_cell(self, background, color, row: int, column: int) -> None:
        cv.rectangle(background,
                     (self.__corner_hor + column * self.SQUARE_SIZE, self.__corner_ver + row * self.SQUARE_SIZE),
                     (self.__corner_hor + (column + 1) * self.SQUARE_SIZE, self.__corner_ver + (row + 1) * self.SQUARE_SIZE),
                     color,
                     -1)

class PlayerDrawer:

    def __init__(self, icon_path: str, hor_pos: int, ver_pos: int):
        self.__corner_hor = hor_pos
        self.__corner_ver = ver_pos
        self.__icon = cv.imread(icon_path, cv.IMREAD_UNCHANGED)

    def draw(self, background, player: PlayerMove) -> None:
        center_pos = self.__draw_center(player.width_pos(), player.height_pos())
        background_part = background[center_pos[1] - self.__icon.shape[0] // 2: center_pos[1] + self.__icon.shape[0] // 2,
                                     center_pos[0] - self.__icon.shape[1] // 2: center_pos[0] + self.__icon.shape[1] // 2]
        alpha_mask = self.__icon[:, :, 3] > 0
        background_part[alpha_mask] = self.__icon[alpha_mask, :3]
        background[center_pos[1] - self.__icon.shape[0] // 2: center_pos[1] + self.__icon.shape[0] // 2,
                   center_pos[0] - self.__icon.shape[1] // 2: center_pos[0] + self.__icon.shape[1] // 2] = background_part

    def __draw_center(self, width: int, height: int) -> tuple:
        return (self.__corner_hor + width * FloorDrawer.SQUARE_SIZE + FloorDrawer.SQUARE_SIZE // 2,
                self.__corner_ver + height * FloorDrawer.SQUARE_SIZE + FloorDrawer.SQUARE_SIZE // 2 )


class TextDrawer:
    FONT = cv.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 1
    FONT_THICKNESS = 2

    def __init__(self, color, hor_pos: int, ver_pos: int):
        self.__corner_hor = hor_pos
        self.__corner_ver = ver_pos
        self.__text_color = color

    def draw_float(self, background, number: float) -> None:
        cv.putText(background, "{:.1f}".format(number), (self.__corner_ver, self.__corner_hor), self.FONT, color=self.__text_color,
                   fontScale=self.FONT_SCALE, thickness=self.FONT_THICKNESS)

    def draw_int(self, background, number: int) -> None:
        cv.putText(background, "{}".format(number), (self.__corner_ver, self.__corner_hor), self.FONT, color=self.__text_color,
                   fontScale=self.FONT_SCALE, thickness=self.FONT_THICKNESS)

    def draw_text(self, background, text: list) -> None:
        total_dy = 0
        FIXED_VERTICAL_DISTANCE = 10
        for line in text:
            (_, dy), _ = cv.getTextSize(line, self.FONT, fontScale=self.FONT_SCALE, thickness=self.FONT_THICKNESS)
            cv.putText(background, line, (self.__corner_ver, self.__corner_hor + total_dy), self.FONT, color=self.__text_color,
                       fontScale=self.FONT_SCALE, thickness=self.FONT_THICKNESS)
            total_dy += dy + FIXED_VERTICAL_DISTANCE



class Screen:
    PLAYER_ICON = "player.png"

    def __init__(self, width: int, height: int, window_name: str):
        self.__player = PlayerDrawer(self.PLAYER_ICON, 100, 50)
        self.__dance_drawer = FloorDrawer(100, 50)
        self.__check_box_drawer = FloorDrawer(200, 400)
        self.__score_drawer = TextDrawer((0, 0, 0), 400, 50)  # Black
        self.__time_drawer = TextDrawer((0, 0, 139), 400, 400)  # Dark Red
        self.__rules_drawer = TextDrawer((0, 0, 0), 50, 50) # Black
        self.__name = window_name

        self.__background = np.zeros((height, width, 3), np.uint8)
        BACKGROUND_COLOR = (169, 169, 169)  # Dark Grey
        for channel in range(3):
            self.__background[:, :, channel] = BACKGROUND_COLOR[channel]

        self.__screen = np.zeros((height, width, 3), np.uint8)

    def draw_initial(self, game_manager: GameManager):
        self.__screen = deepcopy(self.__background)
        self.__dance_drawer.draw(self.__screen, game_manager.dance_floor)
        self.__player.draw(self.__screen, game_manager.player)
        self.__score_drawer.draw_int(self.__screen, game_manager.score)
        self.__time_drawer.draw_float(self.__screen, game_manager.remember_timer.remain())
        cv.imshow(self.__name, self.__screen)

    def draw_guess(self, game_manager: GameManager):
        self.__screen = deepcopy(self.__background)
        self.__dance_drawer.draw(self.__screen, game_manager.empty_dance_floor)
        self.__check_box_drawer.draw(self.__screen, game_manager.check_box)
        self.__player.draw(self.__screen, game_manager.player)
        self.__score_drawer.draw_int(self.__screen, game_manager.score)
        self.__time_drawer.draw_float(self.__screen, game_manager.guess_timer.remain())
        cv.imshow(self.__name, self.__screen)

    def draw_answer(self, game_manager: GameManager):
        self.__screen = self.__get_answer_background(game_manager.is_correct_guess())
        self.__dance_drawer.draw(self.__screen, game_manager.empty_dance_floor)
        self.__check_box_drawer.draw(self.__screen, game_manager.check_box)
        self.__player.draw(self.__screen, game_manager.player)
        self.__score_drawer.draw_int(self.__screen, game_manager.score)
        cv.imshow(self.__name, self.__screen)

    def draw_rules(self):
        RULES = ["Rules:",
                 "1. Remember colors.",
                 "2. Stand on right color.",
                 "3. Repeat.",
                 " ",
                 "Controls:",
                 "WASD/Arrows to move.",
                 "Q/Esc        to qiut.",
                 "R             to restart.",
                 "Any key to start..."]
        self.__screen = deepcopy(self.__background)
        self.__rules_drawer.draw_text(self.__screen, RULES)
        cv.imshow(self.__name, self.__screen)

    def __get_answer_background(self, correct: bool):
        background = deepcopy(self.__background)
        color = (118, 194, 111) if correct else (106, 71, 242)
        for channel in range(3):
            background[:, :, channel] = color[channel]
        return background


if __name__ == "__main__":
    mainscreen = Screen(500, 550, "Window")
    manager = GameManager()
    for level in range(4):
        manager.game_prepare()
        mainscreen.draw_initial(manager)
        cv.waitKey(0)
        manager.player.move_down()
        manager.player.move_right()
        manager.game_choose_step()
        mainscreen.draw_guess(manager)
        cv.waitKey(0)
        manager.game_answer_show_step()
        manager.update_score()
        manager.player.move_right()
        mainscreen.draw_answer(manager)
        cv.waitKey(0)
    cv.destroyAllWindows()
