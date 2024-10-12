from screen import Screen
from game_manager import GameManager
from game_timer import GameTimer
from debug_info import debug_message
import cv2 as cv

class FrameManager:
    FRAMES_PER_SEC = 60
    START_STATE = -1
    REMEMBER_STATE = 0
    GUESS_STATE = 1
    ANSWER_STATE = 2
    LOSE_STATE = 3
    CLOSE_STATE = 4
    KEY_DELAY = 5
    WINDOW_NAME = "Dance Floor Game (Buy now for 1000$)"

    def __init__(self, width: int, height: int):
        self.__screen = Screen(width, height, self.WINDOW_NAME)
        self.__frame_timer = GameTimer(1 / self.FRAMES_PER_SEC)
        self.__current_state = None

    def __init_new_game(self):
        self.__manager = GameManager()
        self.__current_state = self.START_STATE

    def game_frame(self):
        self.__init_new_game()
        while self.__manager.level < 100 and not self.is_exit_state():
            self.__frame_timer.start()
            self.update_manager_by_state()
            self.draw_screen()
            self.process_input()
            self.__frame_timer.wait_till_timer_end()

    def rules_frame(self):
        self.__screen.draw_rules()
        cv.waitKey(0)

    def repeat_required(self):
        if self.__current_state == self.LOSE_STATE or self.__manager.level >= 100:
            key = cv.waitKeyEx(0)
            if key == ord('r') or key == ord('R'):
                return True
        cv.destroyAllWindows()
        return False


    def is_exit_state(self):
        return self.__current_state == self.LOSE_STATE or self.__current_state == self.CLOSE_STATE

    def update_manager_by_state(self):
        if self.__current_state == self.START_STATE:
            self.__manager.game_prepare()
            self.__current_state = self.REMEMBER_STATE
            return
        if self.__current_state == self.REMEMBER_STATE:
            if self.__manager.remember_timer.is_active():
                return
            self.__manager.game_choose_step()
            self.__current_state = self.GUESS_STATE
            return
        if self.__current_state == self.GUESS_STATE:
            if self.__manager.guess_timer.is_active():
                return
            self.__manager.game_answer_show_step()
            self.__current_state = self.ANSWER_STATE
            return
        if self.__current_state == self.ANSWER_STATE:
            if self.__manager.answer_timer.is_active() and self.__manager.is_correct_guess():
                return
            self.__manager.update_score()
            if self.__manager.is_correct_guess():
                self.__manager.game_prepare()
                self.__current_state = self.REMEMBER_STATE
                return
            else:
                self.__current_state = self.LOSE_STATE
        if self.is_exit_state():
            return
        debug_message("Unknown state")
        self.__current_state = self.CLOSE_STATE

    def process_input(self):
        key = cv.waitKeyEx(self.KEY_DELAY)
        if cv.getWindowProperty(self.WINDOW_NAME, cv.WND_PROP_VISIBLE) < 1:
            self.__current_state = self.CLOSE_STATE
            return
        if key < 0:
            return
        if key == ord('w') or key == ord('W') or key == 2490368:
            self.__manager.player.move_up()
        elif key == ord('a') or key == ord('A') or key == 2424832:
            self.__manager.player.move_left()
        elif key == ord('s') or key == ord('S') or key == 2621440:
            self.__manager.player.move_down()
        elif key == ord('d') or key == ord('D') or key == 2555904:
            self.__manager.player.move_right()
        elif key == ord('q') or key == ord('Q') or key == 27:
            self.__current_state = self.CLOSE_STATE



    def draw_screen(self):
        if self.__current_state == self.START_STATE:
            return
        if self.__current_state == self.REMEMBER_STATE:
            self.__screen.draw_initial(self.__manager)
            return
        if self.__current_state == self.GUESS_STATE:
            self.__screen.draw_guess(self.__manager)
            return
        if self.__current_state == self.ANSWER_STATE:
            self.__screen.draw_answer(self.__manager)
            return
        if self.is_exit_state():
            return
        debug_message("Unknown state")
        self.__current_state = self.CLOSE_STATE


if __name__ == "__main__":
    game = FrameManager(500, 550)
    game.game_frame()
