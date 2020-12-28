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
    CAPACITY = 5000
    memory = MemoryReplay(CAPACITY)

    # EPS_START = 0.9
    # EPS_END = 0.05
    # EPS_DECAY = 50
    EPS_THRES = [0.3, 0.1]

    def __init__(self, pos, view_range, map_size):
        self.pos = pos
        self.START_POS = tuple(deepcopy(pos))
        self.view_range = view_range
        self.map_size = map_size
        self.time_step = 0
        self.reward = 0
        # self.cumu_obs = np.array([])
        # self.curr_obs = None
        self.model = DQN(self.memory, view_range)

    def select_action(self, mode, obs):  # mode=0 随机+模型   mode=1 模型  
        sample = np.random.random(1)
        if self.time_step < 100:
            EPS_THRES = self.EPS_THRES[0]
        else:
            EPS_THRES = self.EPS_THRES[1]

        if mode == 1 or sample > EPS_THRES:
                obs = torch.FloatTensor(obs)
                pos = torch.IntTensor(self.pos)
                probs = self.model.get_probs(obs, pos).detach().numpy()
                # print(probs)
                act = np.argmax(probs)
        else:   
            act = np.random.randint(0,4)
        return act

    def store_transition(self, obs, pos, action, next_obs, next_pos, reward):
        return self.memory.push(obs, pos, action, next_obs, next_pos, reward)

    def reset(self):
        self.pos = list(self.START_POS)
        self.time_step = 0
        self.reward = 0
        # self.cumu_obs = np.array([])

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