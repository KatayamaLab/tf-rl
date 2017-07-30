import random
from collections import deque

class ReplayBuffer:

    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def append(self, data):
        self.buffer.append(data)

    def sample(self, size):
        return random.sample(self.buffer, min(len(self.buffer), size))

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
