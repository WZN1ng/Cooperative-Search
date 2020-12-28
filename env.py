import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import animation
from target import Target
import random

# define some colors
COLORS = {'blue':[0,0,1], 'red':[1,0,0], 'white':[1,1,1],
        'black':[0,0,0], 'gray':[0.5,0.5,0.5], 'green':[0,1,0]}

class EnvSearch():
    def __init__(self, map_size, target_num, mode, dict=None):
        self.map_size = map_size
        self.target_num = target_num
        self.target_find = 0
        self.reward_list = {'find_target': 10, 'stay_punish':-50}
        self.total_reward = 0
        self.curr_reward = 0
        # self.explore_num = 0
        self.mode = mode
        self.freq_map = np.zeros((map_size, map_size))   # 频率矩阵
        self.obs_list = []  # 智能体独自当前观测矩阵
        # self.cumu_obs_list = []   # 所有智能体的独自累积观测矩阵
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
        # self.gif = []

        # self.cumulative_joint_obs = np.array([])
        self.land_mark_map = np.zeros((self.map_size, self.map_size)) # initialize map 0：空白  1：目标

        # initialize targets
        self.target_list = []
        self.min_target_pos = self.map_size//4
        self.max_target_pos = 3*self.map_size//4
        self.reset(mode)

    def get_full_obs(self, agent_list):     # 上位机视角全局观测矩阵
        # print('sum', sum(sum(self.land_mark_map)))
        return self.land_mark_map     # 0：空白   1：目标

    def get_agent_obs(self, agent, idx):     # 智能体观测矩阵      0：空白  0.5：未知   1：目标
        obs_size = 2 * agent.view_range - 1
        obs = np.zeros((obs_size, obs_size))
        for i in range(obs_size):
            for j in range(obs_size):
                x = i + agent.pos[0] - agent.view_range + 1     # 坐标转换
                y = j + agent.pos[1] - agent.view_range + 1
                if x >= 0 and x < self.map_size and y >= 0 and y < self.map_size:
                    if self.land_mark_map[x,y] == 1:
                        obs[i,j] = 1   # 目标
                    if (agent.view_range - 1 - i)**2+(agent.view_range - 1 - j)**2 > agent.view_range**2:
                        obs[i,j] = 0.5
                else:   # 超出地图范围  
                    obs[i,j] = 0.5   # 未知区域
                    
        if idx < len(self.obs_list):
            self.obs_list[idx] = obs
        else:
            self.obs_list.append(obs)
        return obs

    def get_current_joint_obs(self, agent_list):    # 当前时刻联合观测矩阵  0：空白  0.5：未知   1：目标
        obs = 0.5*np.ones((self.map_size, self.map_size)) 
        size = 2 * agent_list[0].view_range - 1
        for i, agent in enumerate(agent_list):
            obs_temp = self.obs_list[i]     # 智能体观测矩阵
            posx, posy = agent.pos
            for i in range(size):
                for j in range(size):
                    x = i + posx - agent.view_range + 1
                    y = j + posy - agent.view_range + 1
                    if x >= 0 and x < self.map_size and y >= 0 and y < self.map_size:
                        if obs_temp[i,j] != 0.5:
                            obs[x,y] = obs_temp[i,j]
        return obs
    
    # def get_cumulative_agent_obs(self, idx, agent):   # 单个智能体的累积观测
    #     curr_obs = self.obs_list[idx]     # 获取当前观测
    #     size = 2 * agent.view_range - 1
    #     posx,posy = agent.pos
    #     if agent.cumu_obs.size:
    #         obs = agent.cumu_obs.copy()
    #         for i in range(size):
    #             for j in range(size):
    #                 x = i + posx - agent.view_range + 1
    #                 y = j + posy - agent.view_range + 1
    #                 if x >= 0 and x < self.map_size and y >= 0 and y < self.map_size:
    #                     if curr_obs[i,j] != 0.5 and obs[x,y] == 0.5:
    #                         obs[x,y] = curr_obs[i,j]
    #         agent.cumu_obs = obs        # 将累积观测存入智能体中
    #         # self.cumu_obs_list[i] = obs
    #     else:
    #         obs = 0.5*np.ones((self.map_size, self.map_size))   # 初始化
    #         for i in range(size):
    #             for j in range(size):
    #                 x = i + posx - agent.view_range + 1
    #                 y = j + posy - agent.view_range + 1
    #                 if x >= 0 and x < self.map_size and y >= 0 and y < self.map_size:
    #                     if curr_obs[i,j] != 0.5:
    #                         obs[x,y] = curr_obs[i,j]
    #         agent.cumu_obs = obs
    #         # self.cumu_obs_list.append(obs)  # 加入列表

    def get_cumulative_joint_obs(self, last_obs, agent_list):   # 累积联合观测矩阵   -1：智能体  0：空白   0.5：未知   1：目标   2：路径
        if last_obs.size != 0:
            obs = last_obs.copy()
            count = 0   # 记录新发现的格子数
            size = 2 * agent_list[0].view_range - 1
            for i, agent in enumerate(agent_list):
                obs_temp = self.obs_list[i]
                for i in range(size):
                    for j in range(size):
                        x = i + agent.pos[0] - agent.view_range + 1
                        y = j + agent.pos[1] - agent.view_range + 1
                        if x >= 0 and x < self.map_size and y >= 0 and y < self.map_size:   # 在范围内
                            if obs_temp[i,j] != 0.5 and last_obs[x,y] == 0.5: 
                                obs[x,y] = obs_temp[i,j]
                                count += 1  
                            elif last_obs[x,y] == -1:
                                obs[x,y] = 2   # 标记之前路径
            for agent in agent_list:
                x,y = agent.pos
                obs[x,y] = -1
            self.explore_num = count
            return obs
        else:
            obs = self.get_current_joint_obs(agent_list)
            for agent in agent_list:
                x,y = agent.pos
                obs[x,y] = -1
            return obs

    def get_explore_num(self):
        return self.explore_num

    def save_target(self, filename):    # 保存目标点
        fp = open(filename, 'w')
        for target in self.target_list:
            x, y = target.pos
            fp.write(str(x)+' '+str(y)+'\n')

    def reset(self, mode, dict=None):    # 重设目标位置
        # reset targets
        self.target_list.clear()
        self.land_mark_map = np.zeros((self.map_size, self.map_size))
        # print('mode: ',mode)
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

        # print('sum1', sum(sum(self.land_mark_map)))
        # reset parameter
        self.cumulative_joint_obs = np.array([])
        # self.cumu_obs_list.clear()
        self.obs_list.clear()
        # self.gif.clear()
        self.freq_map = np.zeros((self.map_size, self.map_size))
        self.target_find = 0
        self.total_reward = 0
        self.curr_reward = 0
        self.explore_num = 0

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

    def step(self, agent_list, act_list):     # 环境迭代
        done = False
        info = None
        self.curr_reward = 0

        if len(agent_list) != len(act_list):
            return False

        for i,agent in enumerate(agent_list):
            if agent.time_step == 0:
                x,y = agent.pos
                self.freq_map[x,y] += 1
            self.agent_step(agent, act_list[i])
            x,y = agent.pos
            self.freq_map[x,y] += 1

        for idx,agent in enumerate(agent_list):
            agent.time_step += 1
            obs = self.get_agent_obs(agent, idx)
            size = obs.shape[0]

            for target in self.target_list:
                i, j = target.pos
                x = i - agent.pos[0] + agent.view_range - 1
                y = j - agent.pos[1] + agent.view_range - 1
                if x >= 0 and x < size and y >= 0 and y < size and target.find == False \
                    and obs[x,y] == 1:    # 蓝色
                    self.curr_reward += self.reward_list['find_target']
                    target.find = True
                    self.target_find += 1
        # for i,agent in enumerate(agent_list):
        #     self.get_cumulative_agent_obs(i,agent)
        self.cumulative_joint_obs = self.get_cumulative_joint_obs(self.cumulative_joint_obs, agent_list)    # 更新累积矩阵

        explore_num = self.get_explore_num()
        if explore_num == 0:
            self.curr_reward += self.reward_list['stay_punish']
        x,y = agent_list[0].pos
        self.curr_reward += 1/self.freq_map[x,y]
        self.total_reward += self.curr_reward

        for agent in agent_list:
            agent.reward = self.total_reward       

        if self.target_find == self.target_num:
            done = True
        return done

    def render(self, agent_list):   # 绘图  mode 0 直接显示 mode 1 存gif
        plt.cla()

        # 初始化图像
        img_full_obs = np.ones((self.map_size, self.map_size, 3))
        img_curr_joint_obs = 0.5*np.ones((self.map_size, self.map_size, 3))
        img_cumu_joint_obs = 0.5*np.ones((self.map_size, self.map_size, 3))

        # 获取信息
        full_obs = self.get_full_obs(agent_list)
        curr_joint_obs = self.get_current_joint_obs(agent_list)
        # curr_joint_obs = agent_list[0].cumu_obs  # test
        cumu_joint_obs = self.cumulative_joint_obs

        # processing
        for i in range(self.map_size):
            for j in range(self.map_size):
                # 图1
                if full_obs[i,j] == 1:  # 目标
                    img_full_obs[i,j] = COLORS['blue']

                # 图2
                if curr_joint_obs[i,j] == 1:
                    img_curr_joint_obs[i,j] = COLORS['blue']
                elif curr_joint_obs[i,j] == 0:
                    img_curr_joint_obs[i,j] = COLORS['white']

                # 图3
                if cumu_joint_obs[i,j] == -1:
                    img_cumu_joint_obs[i,j] = COLORS['red']
                elif cumu_joint_obs[i,j] == 0:
                    img_cumu_joint_obs[i,j] = COLORS['white']
                elif cumu_joint_obs[i,j] == 1:
                    img_cumu_joint_obs[i,j] = COLORS['blue']
                elif cumu_joint_obs[i,j] == 2:
                    img_cumu_joint_obs[i,j] = COLORS['green']

        for agent in agent_list:
            x, y = agent.pos
            img_full_obs[x,y] = COLORS['red']
            img_curr_joint_obs[x,y] = COLORS['red']

        self.ax1.imshow(img_full_obs)
        self.ax2.imshow(img_curr_joint_obs)
        self.ax3.imshow(img_cumu_joint_obs)
        title = 'total_reward:{}   target_find:{}/{}'.format(self.total_reward,
                                                                self.target_find,self.target_num)
        plt.suptitle(title)
        
        # if mode == 0:
        plt.draw()
        plt.pause(0.1)  
        # elif mode == 1:
        #     plt.ion()
        #     im = plt.draw()
        #     self.gif.append(im)
    
    # def savegif(self, filename):
    #     plt.cla()
    #     plt.ioff()
    #     fig = plt.figure()
    #     # ani = animation.ArtistAnimation(fig, self.gif, interval=100, repeat_delay=len(self.gif)*100) 
    #     # print(self.gif.__len__())
    #     # # ani.save(filename, writer='pillow')
    #     # plt.show()
    #     for g in self.gif:
    #         plt.draw()
    #         plt.pause(1)