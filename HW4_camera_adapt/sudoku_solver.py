from debug_info import debug_message

CELLS_COUNT = 9
EMPTY_CELL_VALUE = 0

def empty_cell(grid, i, j):
        for x in range(i, CELLS_COUNT):
            for y in range(j, CELLS_COUNT):
                if grid[x][y] == EMPTY_CELL_VALUE:
                        return x, y
        for x in range(CELLS_COUNT):
            for y in range(CELLS_COUNT):
                if grid[x][y] == EMPTY_CELL_VALUE:
                        return x, y
        return None, None

def valid_place(grid, i, j, possible_element):
        valid_row = all([possible_element != grid[i][x] for x in range(CELLS_COUNT)])
        if not valid_row:
            return False

        valid_col = all([possible_element != grid[x][j] for x in range(CELLS_COUNT)])
        if not valid_col:
            return False

        RECT_SIZE = 3
        rec_top_x, rec_top_y = i - i % RECT_SIZE, j - j % RECT_SIZE
        for x in range(RECT_SIZE):
            for y in range(RECT_SIZE):
                if grid[rec_top_x + x][rec_top_y + y] == possible_element:
                    return False
        return True

def solve_iteration(grid, i=0, j=0):
        i,j = empty_cell(grid, i, j)
        if i is None or j is None:
                return True
        for possible_element in range(1, 10):
                if valid_place(grid, i, j, possible_element):
                        grid[i][j] = possible_element
                        if solve_iteration(grid, i, j):
                                return True
                        grid[i][j] = EMPTY_CELL_VALUE
        return False

def check_input_grid(in_grid):
    if in_grid.shape != (CELLS_COUNT, CELLS_COUNT):
        debug_message(f"Input grid shape is {in_grid.shape}, not {(CELLS_COUNT, CELLS_COUNT)}")
        return False

    for x in range(CELLS_COUNT):
        for y in range(CELLS_COUNT):
            if in_grid[x][y] != EMPTY_CELL_VALUE:
                tmp_storage = in_grid[x][y]
                in_grid[x][y] = EMPTY_CELL_VALUE
                valid = True
                if not valid_place(in_grid, x, y, tmp_storage):
                    valid = False
                in_grid[x][y] = tmp_storage
                if not valid:
                    debug_message(f"Cell [{x}, {y}] has incorrect value {tmp_storage} in grid")
                    return False

    return True

def solve(in_grid):
    if not check_input_grid(in_grid):
        return None

    result = in_grid.copy()
    solve_success = solve_iteration(result)

    return result if solve_success else None