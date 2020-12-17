import random
# import json
from collections import namedtuple

class MemoryReplay():
    Transition = namedtuple('Transition',
                        ('obs','pos','action','next_obs','next_pos','reward'))
                        
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):     # 保存经验
        if len(self.memory) == self.capacity:
            return False
        self.memory.append(self.Transition(*args))
        self.position = (self.position+1) % self.capacity
        return True

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