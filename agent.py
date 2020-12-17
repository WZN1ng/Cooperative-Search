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
    CAPACITY = 10000
    memory = MemoryReplay(CAPACITY)

    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200

    def __init__(self, pos, view_range, map_size):
        self.pos = pos
        self.START_POS = tuple(deepcopy(pos))
        self.view_range = view_range
        self.map_size = map_size
        self.time_step = 0
        self.reward = 0

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(view_range)
        self.optimizer = optim.RMSprop(self.model.parameters())
        self.loss_function = nn.MSELoss()

    def select_action(self, obs):
        sample = np.random.random(1)
        eps_threshold = self.EPS_END+(self.EPS_START-self.EPS_END)*np.exp(-1.*self.time_step/self.EPS_DECAY)
        self.time_step += 1

        pos = np.array([x/self.map_size for x in self.pos])
        if sample > eps_threshold:
            # print(obs.shape)
            obs = obs.astype(np.float32)
            obs = torch.from_numpy(obs)
            view = 2*self.view_range-1
            obs = obs.reshape((-1,1,view,view))
            pos = pos.astype(np.float32)
            pos = torch.from_numpy(pos)
            
            prob = self.model(Variable(obs), Variable(pos))
            prob = prob.detach().numpy()
            act = np.argmax(prob)
            # print('prob:{}  act:{}'.format(prob, act))
            return act
        else:
            return np.random.randint(0,4)

    def store_transition(self, obs, pos, action, next_obs, next_pos, reward):
        return self.memory.push(obs, pos, action, next_obs, next_pos, reward)

    def reset(self):
        self.pos = list(self.START_POS)
        self.time_step = 0
        self.reward = 0

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))

    

    