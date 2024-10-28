from screen import screen


if __name__ == "__main__":
    IMAGE1 = "test_pic2.jpg"
    IMAGE2 = "landscape.jpg"
    IMAGE3 = "city.jpg"
    WINDOW = "Fisheye"
    image_screen = screen(WINDOW)
    image_screen.loop(IMAGE1)
