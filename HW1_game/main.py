from frame_manager import FrameManager


if __name__ == "__main__":
    game_frame = FrameManager(500, 550)
    game_frame.rules_frame()
    while True:
        game_frame.game_frame()
        if not game_frame.repeat_required():
            break
