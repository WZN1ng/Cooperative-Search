

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# define some colors
COLORS = {'blue':[0,0,1], 'red':[1,0,0], 'white':[1,1,1],
        'black':[0,0,0], 'gray':[0.5,0.5,0.5], 'green':[0,1,0]}

class Target():
    def __init__(self, pos):
        self.pos = pos 
        self.find = False

class SearchEnv():
    def __init__(self, args, circle_dict=None, targets_filename=None): 
        # env params
        self.map_size = args.map_size
        self.target_num = args.target_num
        self.target_mode = args.target_mode
        self.target_dir = args.target_dir
        self.agent_mode = args.agent_mode
        self.n_agents = args.n_agents
        self.view_range = args.view_range
        self.obs_size = 2*self.view_range-1
        self.circle_dict = circle_dict
        self.targets_filename = targets_filename

        # env variables
        self.time_step = 0
        self.target_list = []
        self.target_find = 0
        self.target_map = np.zeros((self.map_size, self.map_size)) # grid map
        self.agent_pos = []
        self.total_reward = 0
        self.curr_reward = 0
        self.freq_map = np.zeros((self.map_size, self.map_size))    # frequency map
        self.obs = []
        self.state = np.zeros((self.map_size, self.map_size, 2))
        self.cumulative_joint_obs = np.array([])    # joint obs map

        # reward
        self.REWARD_FIND = 10
        self.MOVE_COST = -1

        # show
        self.fig = plt.figure()
        self.gs = GridSpec(1, 3, figure=self.fig)
        self.ax1 = self.fig.add_subplot(self.gs[0:1, 0:1])
        self.ax2 = self.fig.add_subplot(self.gs[0:1, 1:2])
        self.ax3 = self.fig.add_subplot(self.gs[0:1, 2:3])
        self.img_cumu_joint_obs = None

        # init env
        self.reset(init=True)

    # env info
    def get_env_info(self):
        env_info = {}
        env_info['n_actions'] = 4
        env_info['state_shape'] = self.map_size*self.map_size*2
        env_info['obs_shape'] = self.obs_size*self.obs_size + 2     # obs + pos
        env_info['episode_limit'] = 500 
        return env_info

    # reset environment
    def reset(self, init=False):
        if not init:
            self.target_map = np.zeros((self.map_size, self.map_size))
            self.agent_pos.clear()
            self.total_reward = 0
            self.curr_reward = 0
            self.obs.clear()
            self.state = np.zeros((self.map_size, self.map_size, 2))
            self.cumulative_joint_obs = np.array([])
            self.target_list.clear()
            self.time_step = 0
            self.target_find = 0

        # reset targets
        min_target_pos = self.map_size//4
        max_target_pos = 3*self.map_size//4

        if self.target_mode == 0:   # totally random location
            while len(self.target_list) < self.target_num:
                x, y = np.random.randint(0, self.map_size, size=2)
                if self.target_map[x,y] == 0:
                    temp_target = Target([x,y])
                    self.target_list.append(temp_target)
                    self.target_map[x,y] = 1
                    self.state[x,y,0] = 1
        
        elif self.target_mode == 1:    # targets at the edge of the map
            while len(self.target_list) < self.target_num:
                x, y = np.random.randint(0, self.map_size, size=2)
                if self.target_map[x,y] == 0:
                    if x <= min_target_pos or x >= max_target_pos or \
                        y <= min_target_pos or y >= max_target_pos:
                        temp_target = Target([x,y])
                        self.target_list.append(temp_target)
                        self.target_map[x,y] = 1
                        self.state[x,y,0] = 1
                    
        elif self.target_mode == 2:    # high prob of being in the dict circles
            if self.circle_dict:
                for i,(x,y) in enumerate(self.circle_dict['circle_center']):
                    # 中间过程
                    tmp_xl = x-self.circle_dict['circle_radius'][i]
                    tmp_xh = x+self.circle_dict['circle_radius'][i]
                    tmp_yl = y-self.circle_dict['circle_radius'][i]
                    tmp_yh = y+self.circle_dict['circle_radius'][i]
                    curr_num = 0
                    while curr_num < self.circle_dict['target_num'][i]:
                        x = np.random.randint(max(tmp_xl,0), min(tmp_xh,self.map_size))
                        y = np.random.randint(max(tmp_yl,0), min(tmp_yh,self.map_size))
                        if self.target_map[x,y] == 0:
                            temp_target = Target([x,y])
                            self.target_list.append(temp_target)
                            self.target_map[x,y] = 1
                            self.state[x,y,0] = 1
                            curr_num += 1
            else:
                raise Exception('No circle dictionary')
        
        elif self.target_mode == 3:     # load targets from file
            if self.targets_filename:
                fp = open(self.targets_filename, 'r')
                while True:
                    s = fp.readline()
                    if not s:
                        break
                    x, y = [int(x) for x in s.split('\n')[0].split(' ')]
                    temp_target = Target([x,y])
                    self.target_list.append(temp_target)
                    self.target_map[x,y] = 1
                    self.state[x,y,0] = 1
            else:
                raise Exception('No target file')
        
        else:
            raise Exception('Unknown target mode')

        # reset agents' location
        if self.agent_mode == 0:    # reset agents in the center of the map
            length_square = int(np.ceil(np.sqrt(self.n_agents)))
            start_pos = [(self.map_size-length_square)//2 for _ in range(2)]
            for i in range(start_pos[0], start_pos[0]+length_square):
                if len(self.agent_pos) == self.n_agents:
                    break
                for j in range(start_pos[0], start_pos[0]+length_square):
                    if len(self.agent_pos) == self.n_agents:
                        break
                    self.agent_pos.append([i,j])
                    self.freq_map[i,j] += 1
        
        elif self.agent_mode == 1:      # reset agents in the bottom left corner of the map
            length_square = int(np.ceil(np.sqrt(self.n_agents)))
            start_pos = [self.map_size-1, 0]
            for i in range(start_pos[0], start_pos[0]-length_square, -1):
                if len(self.agent_pos) == self.n_agents:
                    break
                for j in range(start_pos[1], start_pos[1]+length_square):
                    if len(self.agent_pos) == self.n_agents:
                        break
                    self.agent_pos.append([i,j])
                    self.freq_map[i,j] += 1

        elif self.agent_mode == 2:      # reset agents in the bottom of the map
            interval = (self.map_size-1)//(self.n_agents-1)
            for i in range(self.n_agents):
                if len(self.agent_pos) == self.n_agents:
                    break
                x = self.map_size-1
                y = i*interval
                self.agent_pos.append([x,y])
                self.freq_map[x,y] += 1
        else:
            raise Exception('Unknown agent mode')

        self._update_obs()
        self._update_state()

    # update state of the env after step
    def get_state(self):
        state = self.state.reshape(2*self.map_size*self.map_size)
        return state

    def _update_state(self):
        self._clear_agent_state()
        for x, y in self.agent_pos:
            self.state[x, y, 1] = 1
    
    def _clear_agent_state(self):
        d = [[0,0],[-1,0],[1,0],[0,1],[0,-1]]
        for x, y in self.agent_pos:
            for dx, dy in d:
                if x+dx >= 0 and x+dx < self.map_size and y+dy >= 0 and y+dy < self.map_size:
                    self.state[x+dx, y+dy, 1] = 0

    # update obs of all agents after step
    def get_obs(self):
        obs = np.array(self.obs)
        pos = np.array(self.agent_pos)
        print(obs.shape, pos.shape)
        obs = obs.reshape(-1, self.obs_size*self.obs_size)
        obs = np.concatenate((obs, pos), axis=1)
        print(obs.shape)
        return obs

    def _update_obs(self):
        self.obs.clear()
        for x, y in self.agent_pos:
            obs = np.zeros((self.obs_size, self.obs_size))
            for i in range(self.obs_size):
                for j in range(self.obs_size):
                    xmap = i + x - self.view_range + 1
                    ymap = j + y - self.view_range + 1
                    if xmap >= 0 and xmap < self.map_size and ymap >= 0 and ymap < self.map_size:
                        if (self.view_range - 1 - i)**2+(self.view_range - 1 - j)**2 > self.view_range**2:
                            obs[i,j] = 0.5
                        elif self.target_map[xmap, ymap] == 1:
                            obs[i,j] = 1
                    else:
                        obs[i,j] = 0.5
            self.obs.append(obs)
    
    # return the one-hot avail action (0:up, 1:left, 2:down, 3:right)
    def get_avail_agent_actions(self, agent_id):
        if agent_id >= self.n_agents:
            raise Exception('Agent id out of range')
        avail_act = np.zeros(4)
        x, y = self.agent_pos[agent_id]
        if x > 0:
            avail_act[0] = 1
        if y > 0:
            avail_act[1] = 1
        if x < self.map_size - 1:
            avail_act[2] = 1
        if y < self.map_size - 1:
            avail_act[3] = 1
        return avail_act

    # env update
    def step(self, act_list):
        if len(act_list) != self.n_agents:
            raise Exception('Act num mismatch agent')
        
        reward = self.MOVE_COST
        terminated = False
        info = ''

        self.time_step += 1 # update timestep
        self._agent_step(act_list)  # agent step

        for i, (x_a, y_a) in enumerate(self.agent_pos):
            # freq reward
            reward += 1/self.freq_map[x_a, y_a]
            # check whether to find new target
            for tar in self.target_list:
                x_t, y_t = tar.pos
                if (x_t-x_a)**2+(y_t-y_a)**2 <= self.view_range**2:
                    if tar.find == False:
                        reward += self.REWARD_FIND
                        tar.find = True
                        self.target_find += 1

        # check whether to finish search
        if self.target_find >= self.target_num:
            terminated = True
        
        # update obs and state
        self._update_obs()
        self._update_state()

        return reward, terminated, info
        
    # agent update
    def _agent_step(self, act):
        if len(act) != self.n_agents:
            raise Exception('Act num mismatch agent')
        for i,(x,y) in enumerate(self.agent_pos):
            if act[i] == 0 and x > 0:      # up
                self.agent_pos[i][0] -= 1
            elif act[i] == 1 and y > 0:     # left
                self.agent_pos[i][1] -= 1
            elif act[i] == 2 and x < self.map_size - 1:     # down
                self.agent_pos[i][0] += 1
            elif act[i] == 3 and y < self.map_size - 1:     # right
                self.agent_pos[i][1] += 1
            else:
                raise Exception('Agent fail to move')
            
            # update freq map
            self.freq_map[self.agent_pos[i][0], self.agent_pos[i][1]] += 1

    # save targets to file
    def save_targets(self, filename):
        root = self.target_dir+filename
        if os.path.exists(root):
            raise Exception('Target file exists')
        fp = open(root, 'w')
        for target in self.target_list:
            x, y = target.pos
            fp.write(str(x)+' '+str(y)+'\n')

    # show process
    def render(self):
        # init imgs
        plt.cla()

        # get info
        img_full_obs = self._get_full_obs()
        img_curr_joint_obs = self._get_current_joint_obs()
        if self.time_step <= 1:
            self.img_cumu_joint_obs = self._get_current_joint_obs()
        else:
            self._update_cumulative_joint_obs()

        self.ax1.imshow(img_full_obs)
        self.ax2.imshow(img_curr_joint_obs)
        self.ax3.imshow(self.img_cumu_joint_obs)
        title = 'target_find:{}/{}'.format(self.target_find, self.target_num)

        # show
        plt.suptitle(title)
        plt.draw()
        if self.target_find == self.target_num:
            plt.pause(3)
        else:
            plt.pause(0.1)

    # get full obs
    def _get_full_obs(self):
        img = np.ones((self.map_size, self.map_size, 3))
        for tar in self.target_list:
            # if tar.find == False:
            x, y = tar.pos
            img[x,y] = COLORS['blue']
        for i,(x,y) in enumerate(self.agent_pos):
            img[x,y] = COLORS['red']
        return img

    # get current joint obs of all agents
    def _get_current_joint_obs(self):
        img = 0.5*np.ones((self.map_size, self.map_size, 3))
        for i,(x,y) in enumerate(self.agent_pos):
            obs = self.obs[i]
            for i in range(self.obs_size):
                for j in range(self.obs_size):
                    imgx = i + x - self.view_range + 1
                    imgy = j + y - self.view_range + 1
                    if imgx >= 0 and imgx < self.map_size and imgy >= 0 and imgy < self.map_size:
                        if obs[i,j] == 0:
                            img[imgx, imgy] = COLORS['white']
                        elif obs[i,j] == 1:
                            img[imgx, imgy] = COLORS['blue']
        for i,(imgx,imgy) in enumerate(self.agent_pos):
            img[imgx, imgy] = COLORS['red']
        return img

    # update cumulative joint obs of all agents
    def _update_cumulative_joint_obs(self):
        if not self.img_cumu_joint_obs.any():
            raise Exception('No cumulative joint obs')

        cumu_obs = self.img_cumu_joint_obs.copy()
        curr_obs = self._get_current_joint_obs()
        for i in range(self.map_size):
            for j in range(self.map_size):
                if curr_obs[i,j,0] != COLORS['gray'][0] \
                    or curr_obs[i,j,1] != COLORS['gray'][1] \
                    or curr_obs[i,j,2] != COLORS['gray'][2]:
                    cumu_obs[i,j] = curr_obs[i,j]
                if self.img_cumu_joint_obs[i,j,0] == COLORS['red'][0] \
                    and self.img_cumu_joint_obs[i,j,1] == COLORS['red'][1] \
                    and self.img_cumu_joint_obs[i,j,2] == COLORS['red'][2]:
                    cumu_obs[i,j] = COLORS['green']
                elif self.img_cumu_joint_obs[i,j,0] == COLORS['green'][0] \
                    and self.img_cumu_joint_obs[i,j,1] == COLORS['green'][1] \
                    and self.img_cumu_joint_obs[i,j,2] == COLORS['green'][2]:
                    cumu_obs[i,j] = COLORS['green']
        for i,(x,y) in enumerate(self.agent_pos):
            cumu_obs[x,y] = COLORS['red']
        self.img_cumu_joint_obs = cumu_obs

    # close env
    def close(self):
        pass