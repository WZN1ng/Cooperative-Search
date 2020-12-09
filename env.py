import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random

class Agent(object):
    def __init__(self, pos, view_range):
        self.pos = pos
        self.view_range = view_range
        self.time_step = 0
        self.reward = 0

class Target(object):
    def __init__(self, pos):
        self.pos = pos

class EnvSearch(object):
    def __init__(self, map_size, agent_num, view_range, target_num):
        self.map_size = map_size
        self.agent_num = agent_num
        self.view_range = view_range
        self.target_num = target_num
        self.reward_list = {'move_cost':-1, 'find_target': 100}
        self.reward_total = 0
        self.time_step = 0
        self.land_mark_map = np.zeros((self.map_size, self.map_size)) # initialize map

        # initialize targets
        self.target_list = []
        min_target_pos = self.map_size//4
        max_target_pos = 3*self.map_size//4
        while len(self.target_list) < self.target_num:
            temp_pos = [random.randint(0, self.map_size-1) for _ in range(2)]
            if (self.land_mark_map[temp_pos[0], temp_pos[1]] == 0) and (temp_pos[0] <= min_target_pos or temp_pos[0] >= max_target_pos) \
                and (temp_pos[1] <= min_target_pos or temp_pos[1] >= max_target_pos):
                temp_target = Target(temp_pos)
                self.land_mark_map[temp_pos[0], temp_pos[1]] == 1 # 目标
                self.target_list.append(temp_target)

        # initialize agents
        length_square = int(np.ceil(np.sqrt(self.agent_num)))
        self.start_pos = [(self.map_size-length_square)//2 for _ in range(2)]
        self.agent_list = []
        for i in range(self.start_pos[0], self.start_pos[0]+length_square):
            if len(self.agent_list) == self.agent_num:
                break
            for j in range(self.start_pos[0], self.start_pos[0]+length_square):
                if len(self.agent_list) == self.agent_num:
                    break
                temp_agent = Agent([i,j], self.view_range)
                self.agent_list.append(temp_agent)

    def get_full_obs(self):     # 全局观测矩阵
        obs = np.ones((self.map_size, self.map_size, 3))
        for target in self.target_list:     # 目标设为蓝色
            obs[target.pos[0], target.pos[1]] = [0, 0, 1]

        for agent in self.agent_list:     # 智能体设为红色
            obs[agent.pos[0], agent.pos[1]] = [1, 0, 0]
        return obs      

    def get_agent_obs(self, agent):     # 智能体观测矩阵
        obs_size = 2 * agent.view_range - 1
        obs = np.ones((obs_size, obs_size, 3))
        for i in range(obs_size):
            for j in range(obs_size):
                x = i + agent.pos[0] - agent.view_range + 1
                y = j + agent.pos[1] - agent.view_range + 1

                for k in range(self.target_num):
                    if self.target_list[k].pos[0] == x and self.target_list[k].pos[1] == y:
                        obs[i, j] = [0, 0, 1]

                if i == agent.view_range-1 and j == agent.view_range-1:
                    obs[i, j] = [1, 0, 0]

                if (agent.view_range - 1 - i)*(agent.view_range - 1 - i)+(agent.view_range - 1 - j)*(agent.view_range - 1 - j) > agent.view_range*agent.view_range:
                    obs[i, j] = [0.5, 0.5, 0.5]
        return obs

    def get_current_joint_obs(self):    # 当前时刻联合观测矩阵
        obs = 0.5*np.ones((self.map_size, self.map_size, 3))
        for agent in self.agent_list:
            temp = self.get_agent_obs(agent)
            size = temp.shape[0]
            for i in range(size):
                for j in range(size):
                    x = i + agent.pos[0] - agent.view_range + 1
                    y = j + agent.pos[1] - agent.view_range + 1
                    if x >= 0 and x < self.map_size and y >= 0 and y < self.map_size:
                        if temp[i, j, 0] != 0.5: 
                            obs[x, y] = temp[i, j]
        for agent in self.agent_list:
            [x,y] = agent.pos
            obs[x, y] = [1, 0, 0]
        return obs

    def get_cumulative_joint_obs(self, last_obs):   # 累积联合观测矩阵
        if last_obs.size != 0:
            obs = last_obs.copy()
            for agent in self.agent_list:
                temp = self.get_agent_obs(agent)
                size = temp.shape[0]
                for i in range(size):
                    for j in range(size):
                        x = i + agent.pos[0] - agent.view_range + 1
                        y = j + agent.pos[1] - agent.view_range + 1
                        if x >= 0 and x < self.map_size and y >= 0 and y < self.map_size:
                            if temp[i, j, 0] != 0.5 and last_obs[x, y, 0] == 0.5:
                                obs[x, y] = temp[i, j]
                            if last_obs[x, y, 0] == 1 and last_obs[x, y, 1] == 0 and last_obs[x, y, 2] == 0:
                                obs[x, y] = [0, 1, 0]   # 将之前的路径涂成绿色

            for agent in self.agent_list:
                [x,y] = agent.pos
                obs[x, y] = [1, 0, 0]
            return obs
        else:
            return self.get_current_joint_obs()

    def rand_reset_target_pos(self):    # 重设目标位置
        for k in range(self.target_num):
            self.target_list[k].pos = [random.randint(0, self.map_size-1) for _ in range(2)]

    def agent_step(self, agent_act_list):   # 智能体行动
        if len(agent_act_list) != self.agent_num:
            return
        for k in range(self.agent_num):
            if agent_act_list[k] == 0:      # 向上
                if self.agent_list[k].pos[0] > 0:
                    self.agent_list[k].pos[0] = self.agent_list[k].pos[0] - 1
            elif agent_act_list[k] == 1:    # 向下
                if self.agent_list[k].pos[0] < self.map_size - 1:
                    self.agent_list[k].pos[0] = self.agent_list[k].pos[0] + 1
            elif agent_act_list[k] == 2:    # 向左
                if self.agent_list[k].pos[1] > 0:
                    self.agent_list[k].pos[1] = self.agent_list[k].pos[1] - 1
            elif agent_act_list[k] == 3:    # 向右
                if self.agent_list[k].pos[1] < self.map_size - 1:
                    self.agent_list[k].pos[1] = self.agent_list[k].pos[1] + 1

    def step(self, agent_act_list):     # 环境迭代
        self.agent_step(agent_act_list)
        
