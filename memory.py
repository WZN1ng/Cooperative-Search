import random
# import json
from collections import namedtuple

class MemoryReplay():
    # Transition = namedtuple('Transition',
    #                     ('state','action','next_state','reward'))
                        
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward):     # 保存经验   若已满则按顺序覆盖
        if len(self.memory) == self.capacity:
            self.memory[self.position] = [state, action, next_state, reward]
        else:
            self.memory.append([state, action, next_state, reward])
        self.position = (self.position+1) % self.capacity

    def sample(self, batch_size):   # 抽取经验
        return random.sample(self.memory, batch_size)

    # def filestore(self, filename):      # 写入文件
    #     fp = open(filename, 'w')
    #     for mem in self.memory:
    #         j = json.dumps(mem._asdict())
    #         fp.write(j)
    #         print(j)

    def __len__(self):
        return len(self.memory)

    def is_full(self):
        if len(self.memory) == self.capacity:
            return True
        else:
            return False