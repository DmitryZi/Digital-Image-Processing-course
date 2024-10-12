from debug_info import debug_message
from copy import deepcopy
from random import shuffle

class DanceFloor:
    MIN_WIDTH = 1
    MIN_HEIGHT = 1
    MAX_WIDTH = 4
    MAX_HEIGHT = 4

    def __init__(self, width: int, height: int) -> None:
        if not (self.MIN_WIDTH <= width <= self.MAX_WIDTH):
            debug_message("Width {} is not in bounds from {} to {}".format(width, self.MIN_WIDTH, self.MAX_WIDTH))
        if not (self.MIN_HEIGHT <= height <= self.MAX_HEIGHT):
            debug_message("Height {} is not in bounds from {} to {}".format(height, self.MIN_HEIGHT, self.MAX_HEIGHT))
        self.__width = min(max(self.MIN_WIDTH, width), self.MAX_WIDTH)
        self.__height = min(max(self.MIN_HEIGHT, height), self.MAX_HEIGHT)
        self.clear_colors()
    
    def clear_colors(self) -> None:
        self.__colors = [0]
        self.__colors_mapping = [0] * self.get_cells_count()

    @staticmethod
    def __area(width: int, height: int) -> int:
        return width * height
    
    def __get_cell_number(self, row: int, column: int) -> int:
        if not (0 <= row <= self.__height) or not (0 <= column <= self.__width):
            debug_message(f"cell with row {row} and column {column} is out of bounds [0,{self.__height}], [0, {self.__width}")
            return 0
        return row * self.__width + column
    
    @staticmethod
    def __get_chunk_size(chunk_num: int, chunks_count: int, total_size: int) -> int:
        if chunks_count > 0 and 0 <= chunk_num < chunks_count:
            return (total_size + chunks_count - 1 - chunk_num) // chunks_count
        return 0

    @staticmethod
    def __split_into_chunks(chunks_count: int, list_size: int) -> list:
        result = []
        for chunk_num in range(chunks_count):
            chunk_size = DanceFloor.__get_chunk_size(chunk_num, chunks_count, list_size)
            result.extend([chunk_num] * chunk_size)
        return result
    
    def get_cells_count(self) -> int:
        return DanceFloor.__area(self.__width, self.__height)

    def get_height(self) -> int:
        return self.__height

    def get_width(self) -> int:
        return self.__width

    def fill_dance_floor(self, colors: list) -> None:
        min_cells_count = DanceFloor.__area(self.MIN_WIDTH, self.MIN_HEIGHT)
        max_cells_count = DanceFloor.__area(self.MAX_WIDTH, self.MAX_HEIGHT)
        colors_count = len(colors)
        if not (min_cells_count <= colors_count <= max_cells_count):
            debug_message(f"Invalid amount of colors {colors_count}: from {min_cells_count} to {max_cells_count} is required")
            self.clear_colors()
            return
        self.__colors = deepcopy(colors)
        self.__colors_mapping = self.__split_into_chunks(colors_count, self.get_cells_count())
        shuffle(self.__colors_mapping)
    
    def get_cell_color(self, row: int, column: int):
        return self.__colors[self.__colors_mapping[self.__get_cell_number(row, column)]]
    
    def set_cell_color(self, color, row: int, column: int):
        if not color in self.__colors:
            self.__colors.append(color)
        self.__colors_mapping[self.__get_cell_number(row, column)] = self.__colors.index(color)

    def print(self) -> None:
        print()
        for row in range(self.__height):
            for column in range(self.__width):
                print(self.get_cell_color(row, column), end=' ')
            print()

if __name__ == "__main__":
    # Test
    dance = DanceFloor(3, 3)
    colors = ["Red   ", "Yellow", "Blue  "]
    dance.fill_dance_floor(colors)
    dance.print()
    dance.fill_dance_floor(colors)
    dance.print()

