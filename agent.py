from dqn import DQN
from memory import MemoryReplay

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from copy import deepcopy

class Agent():
    CAPACITY = 2000
    memory = MemoryReplay(CAPACITY)

    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 50
    EPS_THRES = 0.2

    def __init__(self, pos, view_range, map_size):
        self.pos = pos
        self.START_POS = tuple(deepcopy(pos))
        self.view_range = view_range
        self.map_size = map_size
        self.time_step = 0
        self.reward = 0
        self.cumu_obs = np.array([])
        # self.curr_obs = None
        self.model = DQN(self.memory)

    def select_action(self, mode):  # mode=0 随机+模型   mode=1 模型
        if self.cumu_obs.size == 0:
            act = np.random.randint(0,4)
        elif mode == 0:   # 随机探索和模型结合
            sample = np.random.random(1)
            # eps_threshold = self.EPS_END+(self.EPS_START-self.EPS_END)*np.exp(-1.*self.time_step/self.EPS_DECAY)
            if sample > self.EPS_THRES: # mode = 1模型
                probs = self.model.get_probs(torch.FloatTensor(self.cumu_obs)).detach().numpy()
                act = np.argmax(probs)
                # print(probs)
            else:
                act = np.random.randint(0,4)
        else:   # mode == 1
            probs = self.model.get_probs(torch.FloatTensor(self.cumu_obs)).detach().numpy()
            act = np.argmax(probs)
            # print(probs)
        return act

    def store_transition(self, state, action, next_state, reward):
        return self.memory.push(state, action, next_state, reward)

    def reset(self):
        self.pos = list(self.START_POS)
        self.time_step = 0
        self.reward = 0
        self.cumu_obs = np.array([])

    def save_model(self, fileroot1, fileroot2):
        self.model.save_state_dict(fileroot1, fileroot2)

    def load_model(self, fileroot1, fileroot2):
        self.model.load_state_dict(fileroot1, fileroot2)

if __name__ == "__main__":
     a = Agent([23,24], 6, 50)
     for _ in range(50):
        a.store_transition(np.ones((50,50)), 1, np.zeros((50,50)), -10)
     s = a.memory.sample(3)
     batch = a.memory.Transition(*zip(*s))
    #  print(batch.action)