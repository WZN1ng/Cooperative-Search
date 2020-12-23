import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from target import Target
import random

# define some colors
COLORS = {'blue':[0,0,1], 'red':[1,0,0], 'white':[1,1,1],
        'black':[1,1,1], 'gray':[0.5,0.5,0.5], 'green':[0,1,0]}

class EnvSearch():
    def __init__(self, map_size, target_num, mode, dict=None):
        self.map_size = map_size
        self.target_num = target_num
        self.target_find = 0
        self.reward_list = {'move_cost':-1, 'find_target': 100, 'wall_punish':-5, 'explore_reward':0.1}
        self.mode = mode
        self.obs_list = []
        if dict:    
        # mode 2:  circle_num:int, target_empty:[x,y], circle_center:[[x,y],...],circle_radius:[...], target_num:[...]
        # mode 3:  filename
            self.dict = dict

        # show process
        self.fig = plt.figure()
        self.gs = GridSpec(1, 3, figure=self.fig)
        self.ax1 = self.fig.add_subplot(self.gs[0:1, 0:1])
        self.ax2 = self.fig.add_subplot(self.gs[0:1, 1:2])
        self.ax3 = self.fig.add_subplot(self.gs[0:1, 2:3])

        # self.cumulative_joint_obs = np.array([])
        self.land_mark_map = np.zeros((self.map_size, self.map_size)) # initialize map

        # initialize targets
        self.target_list = []
        self.min_target_pos = self.map_size//4
        self.max_target_pos = 3*self.map_size//4
        self.reset(mode)

    def get_full_obs(self, agent_list):     # 上位机视角全局观测矩阵
        show = np.ones((self.map_size, self.map_size, 3))
        for target in self.target_list:
            i,j = target.pos
            show[i,j] = COLORS['blue']
        for agent in agent_list:
            i,j = agent.pos
            show[i,j] = COLORS['red']
        return show

    def get_agent_obs(self, agent, idx):     # 智能体观测矩阵
        obs_size = 2 * agent.view_range - 1
        obs = np.zeros((obs_size, obs_size))
        count = 0
        for i in range(obs_size):
            for j in range(obs_size):
                x = i + agent.pos[0] - agent.view_range + 1
                y = j + agent.pos[1] - agent.view_range + 1
                if x >= 0 and x < self.map_size and y >= 0 and y < self.map_size:
                    if self.land_mark_map[x,y] == 1:
                        obs[i,j] = 1   # 目标
                    elif self.land_mark_map[x,y] == 0:
                        self.land_mark_map[x,y] = 
                
                elif (agent.view_range - 1 - i)**2+(agent.view_range - 1 - j)**2 > agent.view_range**2:
                    obs[i,j] = 0.5   # 未知区域
                    
        if idx < len(self.obs_list):
            self.obs_list[idx] = obs
        else:
            self.obs_list.append(obs)
        return obs

    def get_current_joint_obs(self, agent_list):    # 当前时刻联合观测矩阵
        show = 0.5*np.ones((self.map_size, self.map_size, 3))
        for idx, agent in enumerate(agent_list):
            temp = self.obs_list[idx]
            size = temp.shape[0]
            posx, posy = agent.pos
            for i in range(size):
                for j in range(size):
                    x = i + posx - agent.view_range + 1
                    y = j + posy - agent.view_range + 1
                    if x >= 0 and x < self.map_size and y >= 0 and y < self.map_size:
                        if temp[i,j] != 0.5:
                            show[x,y] = COLORS['white']
                        elif temp[i,j] == 1:
                            show[x,y] = COLORS['blue']

        for agent in agent_list:
            x, y = agent.pos
            show[x,y] = COLORS['red']
        return show

    def get_cumulative_joint_obs(self, last_show, agent_list):   # 累积联合观测矩阵
        if last_show.size != 0:
            show = last_show.copy()
            for idx, agent in enumerate(agent_list):
                temp = self.obs_list[idx]
                size = temp.shape[0]
                for i in range(size):
                    for j in range(size):
                        x = i + agent.pos[0] - agent.view_range + 1
                        y = j + agent.pos[1] - agent.view_range + 1
                        if x >= 0 and x < self.map_size and y >= 0 and y < self.map_size:
                            if temp[i,j] != 0.5 and last_show[x,y,0] == COLORS['gray'][0] and last_show[x,y,1] == COLORS['gray'][1] \
                                and last_show[x,y,2] == COLORS['gray'][2]:
                                if temp[i,j] == 0:
                                    show[x,y] = COLORS['white']
                                elif temp[i,j] == 1:
                                    show[x,y] = COLORS['blue']
                            elif last_show[x,y,0] == COLORS['red'][0] and last_show[x,y,1] == COLORS['red'][1] \
                                and last_show[x,y,2] == COLORS['red'][2]:
                                show[x,y] = COLORS['green']   # 将之前的路径涂成绿色
            for agent in agent_list:
                x, y = agent.pos
                show[x,y] = COLORS['red']
            return show
        else:
            return self.get_current_joint_obs(agent_list)

    def save_target(self, filename):    # 保存目标点
        fp = open(filename, 'w')
        for target in self.target_list:
            x, y = target.pos
            fp.write(str(x)+' '+str(y)+'\n')

    def reset(self, mode, dict=None):    # 重设目标位置
        # reset targets
        self.target_list.clear()

        if mode == 0:   # 目标完全随机分布
            while len(self.target_list) < self.target_num:
                x, y = [random.randint(0, self.map_size-1) for _ in range(2)]
                temp_target = Target([x,y])
                self.target_list.append(temp_target)
                self.land_mark_map[x,y] = 1 # 目标

        elif mode == 1:   # 目标分布在地图边缘
            while len(self.target_list) < self.target_num:
                x, y = [random.randint(0, self.map_size-1) for _ in range(2)]
                if self.land_mark_map[x,y] == 0:
                    if x <= self.min_target_pos or x >= self.max_target_pos or \
                        y <= self.min_target_pos or y >= self.max_target_pos:
                        temp_target = Target([x,y])
                        self.target_list.append(temp_target)
                        self.land_mark_map[x,y] = 1 # 目标

        elif mode == 2:     # 更大概率分布在指定园内
            for i,(x,y) in enumerate(self.dict['circle_center']):
                # 中间过程
                tmp_xl = x-self.dict['circle_radius'][i]
                tmp_xh = x+self.dict['circle_radius'][i]
                tmp_yl = y-self.dict['circle_radius'][i]
                tmp_yh = y+self.dict['circle_radius'][i]
                for _ in range(self.dict['target_num'][i]):
                    pos_x = random.randint(max(tmp_xl,0), min(tmp_xh,self.map_size-1))
                    pos_y = random.randint(max(tmp_yl,0), min(tmp_yh,self.map_size-1))
                    temp_target = Target([pos_x,pos_y])
                    self.target_list.append(temp_target)
                    self.land_mark_map[pos_x,pos_y] = 1
        
        elif mode == 3:     # 载入固定目标
            filename = self.dict['filename']
            fp = open(filename, 'r')
            while True:
                s = fp.readline()
                if not s:
                    break
                x, y = [int(x) for x in s.split('\n')[0].split(' ')]
                temp_target = Target([x,y])
                self.target_list.append(temp_target)
                self.land_mark_map[x,y] = 1

        # reset parameter
        self.cumulative_joint_obs = np.array([])
        self.target_find = 0

    def agent_step(self, agent, act):   # 智能体行动
        if act == 0 and agent.pos[0] > 0:      # 向上
            agent.pos[0] -= 1
            return True
        elif act == 1 and agent.pos[1] > 0:     # 向左
            agent.pos[1] -= 1
            return True
        elif act == 2 and agent.pos[0] < self.map_size - 1:     # 向下
            agent.pos[0] += 1
            return True
        elif act == 3 and agent.pos[1] < self.map_size - 1:     # 向右
            agent.pos[1] += 1
            return True
        else:
            return False # 撞墙 移动失败

    def step(self, idx, agent, act):     # 环境迭代
        done = False
        info = None
        reward = self.reward_list['move_cost']

        if not self.agent_step(agent, act):
            reward += self.reward_list['wall_punish']

        agent.time_step += 1
        obs = self.get_agent_obs(agent, idx)
        size = obs.shape[0]

        for target in self.target_list:
            i, j = target.pos
            x = i - agent.pos[0] + agent.view_range - 1
            y = j - agent.pos[1] + agent.view_range - 1
            if x >= 0 and x < size and y >= 0 and y < size and target.find == False \
                and obs[x,y] == 1:    # 蓝色
                reward += self.reward_list['find_target']
                target.find = True
                self.target_find += 1
        
        # 给予阶段性奖励，鼓励探索


        if self.target_find == self.target_num:
            done = True
        return obs, reward, done, info

    def render(self, agent_list, total_reward=None):   # 绘图
        plt.cla()
        self.ax1.imshow(self.get_full_obs(agent_list))
        self.ax2.imshow(self.get_current_joint_obs(agent_list))
        self.cumulative_joint_obs = self.get_cumulative_joint_obs(self.cumulative_joint_obs, agent_list)
        self.ax3.imshow(self.cumulative_joint_obs)
        if total_reward:
            title = 'total_reward: {}   target_find:{}/{}'.format(str('%.f'%total_reward),
                                                                self.target_find,self.target_num)
            plt.suptitle(title)
        plt.draw()
        plt.pause(0.1)