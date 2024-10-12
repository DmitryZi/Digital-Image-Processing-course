

class PlayerMove:

    def __init__(self, max_width: int, max_height: int) -> None:
        self.__max_width = max_width
        self.__max_height = max_height
        self.__width_pos = self.__max_width // 2
        self.__height_pos = self.__max_height // 2
        self.__can_move = True

    def move_left(self):
        if self.__can_move:
            self.__width_pos = max(0, self.__width_pos - 1)

    def move_right(self):
        if self.__can_move:
            self.__width_pos = min(self.__max_width - 1, self.__width_pos + 1)

    def move_up(self):
        if self.__can_move:
            self.__height_pos = max(0, self.__height_pos - 1)

    def move_down(self):
        if self.__can_move:
            self.__height_pos = min(self.__max_height - 1, self.__height_pos + 1)

    def stop(self):
        self.__can_move = False

    def enable_move(self):
        self.__can_move = True

    def width_pos(self) -> int:
        return self.__width_pos

    def height_pos(self) -> int:
        return self.__height_pos


if __name__ == "__main__":
    move = PlayerMove(3, 2)
    print(move.width_pos (), move.height_pos ())
    move.move_down()
    move.move_down()
    move.move_down()
    print(move.width_pos (), move.height_pos ())
    move.stop()
    move.move_left()
    print(move.width_pos (), move.height_pos ())
