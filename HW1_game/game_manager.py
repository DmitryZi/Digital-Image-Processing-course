from dance_floor_data import DanceFloor
from player_move import PlayerMove
from game_timer import  GameTimer
from random import choice

class GameManager:
    COLORS = [(0, 0, 255),      # Red
              (0, 255, 255),    # Yellow
              (0, 255, 0),      # Green
              (255, 0, 0),      # Blue
              (128, 128, 128),  # Grey
              (19, 69, 139),    # Brown
              (238, 130, 238),  # Purple
              (255, 191, 128),  # Light Blue
              (0, 0, 0)]        # Black
    EMPTY_COLOR = [(255, 255, 255)]
    DANCE_FLOOR_WIDTH = 3
    DANCE_FLOOR_HEIGHT = 3
    MOVE_DELAY = 0.03
    REMEMBER_DELAY = 4
    GUESS_DELAY = 3
    ANSWER_SHOW = 1.5

    def __init__(self):
        self.dance_floor = DanceFloor(self.DANCE_FLOOR_WIDTH, self.DANCE_FLOOR_HEIGHT)
        self.player = PlayerMove(self.DANCE_FLOOR_WIDTH, self.DANCE_FLOOR_HEIGHT)
        self.check_box = DanceFloor(1, 1)
        self.round_colors = self.EMPTY_COLOR
        self.empty_dance_floor = DanceFloor(self.DANCE_FLOOR_WIDTH, self.DANCE_FLOOR_HEIGHT)
        self.move_timer = GameTimer(self.MOVE_DELAY)
        self.remember_timer = GameTimer(self.REMEMBER_DELAY)
        self.guess_timer = GameTimer(self.GUESS_DELAY)
        self.answer_timer = GameTimer(self.ANSWER_SHOW)
        self.score = 0
        self.level = 0

    def game_prepare(self):
        self.round_colors = self.COLORS[:self.__colors_count(self.level)]
        self.dance_floor.fill_dance_floor(self.round_colors)
        self.empty_dance_floor.fill_dance_floor(self.EMPTY_COLOR)
        self.remember_timer.start()
        self.player.enable_move()

    def __colors_count(self, round_num: int) -> int:
        MIN_COLORS = min(3, len(self.COLORS))
        return MIN_COLORS + max(min(len(self.COLORS) - MIN_COLORS, round_num // 2), 0)

    def choose_guess_color(self):
        return choice([level_color for level_color in self.round_colors if level_color != self.check_box.get_cell_color(0, 0)])

    def game_choose_step(self):
        self.check_box.fill_dance_floor([self.choose_guess_color()])
        self.guess_timer.start()

    def game_answer_show_step(self):
        self.player.stop()
        guessed_color = self.check_box.get_cell_color(0, 0)
        for row in range(self.empty_dance_floor.get_height()):
            for column in range(self.empty_dance_floor.get_width()):
                if guessed_color == self.dance_floor.get_cell_color(row, column):
                    self.empty_dance_floor.set_cell_color(guessed_color, row, column)
        self.answer_timer.start()

    def is_correct_guess(self):
        return self.check_box.get_cell_color(0, 0) == self.dance_floor.get_cell_color(self.player.height_pos(), self.player.width_pos())

    def update_score(self):
        if self.is_correct_guess():
            self.level += 1
            self.score += 1


if __name__ == "__main__":
    # Test
    gamemanager = GameManager()
    for level in range(5):
        print(f"ROUND {level}")
        gamemanager.game_prepare()
        gamemanager.dance_floor.print()
        gamemanager.check_box.print()
        gamemanager.game_choose_step()
        gamemanager.dance_floor.print()
        gamemanager.check_box.print()
