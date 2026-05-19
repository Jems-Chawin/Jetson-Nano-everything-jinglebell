import time

class GETFPS:
    def __init__(self, stream_id):
        self.start_time = time.time()
        self.is_first = True
        self.frame_count = 0
        self.stream_id = stream_id

    def get_fps(self):
        end_time = time.time()
        if self.is_first:
            self.start_time = end_time
            self.is_first = False
        if end_time - self.start_time > 5:
            print("FPS of stream %d: %.1f" % (self.stream_id, self.frame_count / 5.0))
            self.frame_count = 0
            self.start_time = end_time
        else:
            self.frame_count += 1
