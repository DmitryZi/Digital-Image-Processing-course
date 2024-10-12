from time import time, sleep

class GameTimer:

    def __init__(self, seconds: float):
        self.__seconds = seconds
        self.__timestamp = None

    def start(self):
        self.__timestamp = time()

    def progress(self):
        return max(min(1, (time() - self.__timestamp) / self.__seconds), 0)

    def is_active(self):
        return self.progress() < 1

    def remain(self):
        return max(0, self.__seconds - (time() - self.__timestamp))

    def wait_till_timer_end(self):
        sleep(self.remain())


if __name__ == "__main__":
    # Test
    timer = GameTimer(5)

    for step in range(5):
        timer.start()
        sleep(step / 2)
        print(timer.progress())
        while timer.is_active():
            sleep(0.5)
            print(timer.progress())

    print("END CHECK")
    timer.start()
    print("STARTED")
    timer.wait_till_timer_end()
    if timer.is_active():
        print("STILL ACTIVE")
