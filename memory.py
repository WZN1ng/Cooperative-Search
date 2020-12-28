import random
# import json
from collections import namedtuple

class MemoryReplay():   
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, obs, pos, action, next_obs, next_pos, reward):     # 保存经验   若已满则按顺序覆盖
        if len(self.memory) == self.capacity:
            self.memory[self.position] = [obs, pos, action, next_obs, next_pos, reward]
        else:
            self.memory.append([obs, pos, action, next_obs, next_pos, reward])
        self.position = (self.position+1) % self.capacity

    def sample(self, batch_size):   # 抽取经验
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def is_full(self):
        if len(self.memory) == self.capacity:
            return True
        else:
            return False