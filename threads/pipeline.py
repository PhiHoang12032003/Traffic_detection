import queue

class Pipeline(queue.Queue):
    def __init__(self, maxsize=256):
        super().__init__(maxsize=maxsize)

    def get_message(self, block=True):
        try:
            return self.get(block=block)
        except queue.Empty:
            return None

    def set_message(self, mess):
        self.put(mess)
