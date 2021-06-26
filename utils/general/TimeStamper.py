#!/usr/bin/env python3

import time


class TimeStamper:
    def __init__(self):
        self.time = time
        self.id = -1
        self.start = time.time()
        self.prev = self.start

    def __call__(self, comment):
        now = self.time.time()
        from_start = now - self.start

        self.from_start_hhmmss = self.time.strftime(
            "%H:%M:%S", self.time.gmtime(from_start)
        )
        from_prev = now - self.prev

        self.from_prev_hhmmss = self.time.strftime(
            "%H:%M:%S", self.time.gmtime(from_prev)
        )

        self.id += 1
        self.prev = now

        print(
            "Time (id:{}): tot {}, prev {} [hh:mm:ss]: {}\n".format(
                self.id, self.from_start_hhmmss, self.from_prev_hhmmss, comment
            )
        )

    def get(self):
        return self.id, self.from_start_hhmmss, self.from_prev_hhmmss, comment
