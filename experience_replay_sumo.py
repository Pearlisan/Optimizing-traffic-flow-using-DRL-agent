# ============================================================
# BLOCK 1: Import libraries
# ============================================================
import random
from collections import deque


# ============================================================
# BLOCK 2: Define replay memory
# ============================================================
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)