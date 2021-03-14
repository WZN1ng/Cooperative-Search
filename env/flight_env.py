import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

class Target():
    def __init__(self, pos, priority):
        self.pos = pos
        self.priority = priority
        self.find = False

class FlightSearchEnv():
    def __init__(self, args, circle_dict):
        # env params 
        self.map_size = args.map_size
        self.target_num = args.target_num
        self.agent_mode = args.agent_mode
        self.n_agents = args.n_agents
        self.view_range = args.view_range
        self.circle_dict = circle_dict
        self.time_limit = args.time_limit
        self.turn_limit = args.turn_limit
        self.detect_prob = args.detect_prob
        self.safe_dist = args.safe_dist
        self.velocity = args.agent_velocity
        self.force_dist = args.force_dist
        self.n_actions = 3
        self.state_shape = self.n_agents*3 + self.target_num*3 # agent: x,y,yaw target: x,y,find
        self.obs_shape = 4 # x,y,yaw,find_num

        # env variables
        self.time_step = 0
        self.target_list = []
        self.target_pos = []
        self.target_find = 0
        self.agent_pos = []
        self.agent_yaw = []
        self.total_reward = 0
        self.curr_reward = 0
        self.obs = []
        self.freq_map = np.zeros((self.map_size+1, self.map_size+1))
        self.out_flag = []          # flag = 1 only when agent get out of the map

        # reward
        self.MOVE_COST = -0.01
        self.REWARD_FACTOR = 20
        # self.COLLI_PUNISH = -1
        self.OUT_PUNISH = -0.5

        # potentail force factor
        self.POTENTIAL_FORCE_FACTOR = 0.8

        # init env
        self.reset(init=True)
        self.get_state()

    def get_env_info(self):
       env_info = {}
       env_info['n_actions'] = self.n_actions
       env_info['state_shape'] = self.state_shape    
       env_info['obs_shape'] = self.obs_shape    
       env_info['episode_limit'] = self.time_limit
       return env_info
       
    def reset(self, init=False):
        if not init:
            self.time_step = 0
            self.target_find = 0
            self.total_reward = 0
            self.curr_reward = 0
            self.obs.clear()
        self.target_list.clear()
        self.target_pos.clear()
        self.agent_pos.clear()
        self.agent_yaw.clear()
        self.out_flag.clear()

        # reset targets 
        for i in range(self.target_num):
            a = 10
            x = a*self.circle_dict['x'][i]
            y = a*self.circle_dict['y'][i]
            deter = self.circle_dict['deter'][i]
            priority = self.circle_dict['priority'][i]
            dx = a*self.circle_dict['dx'][i]
            dy = a*self.circle_dict['dy'][i]
            if deter == 't':
                target_tmp = Target([x,y], priority)
            elif deter == 'f':
                delta_x = dx*2*(np.random.randn()-0.5)
                delta_y = dy*2*(np.random.randn()-0.5)
                x += delta_x
                y += delta_y
                target_tmp = Target([x,y], priority)
            self.target_list.append(target_tmp)
            self.target_pos.append([x,y])

        # reset agents
        if self.agent_mode == 0:    # start from the bottom of the map
            n = self.n_agents // 2
            start_x = self.map_size//2-n*self.safe_dist
            for i in range(self.n_agents):
                x = start_x + i*self.safe_dist
                y = 0
                self.agent_pos.append([x,y])
                self.agent_yaw.append(np.pi/2)
                self.out_flag.append(0)
        elif self.agent_mode == 1: # start from the bottom line(left corner, midium, right corner)
            x = [0, self.map_size/2, self.map_size]
            y = 0
            for i in range(self.n_agents):
                self.agent_pos.append([x[i],y])
                self.agent_yaw.append(np.pi/2)
                self.out_flag.append(0)
        elif self.agent_mode == 2:  # start from the midium of the map
            x = [self.map_size/4, self.map_size/2, self.map_size*3/4]
            y = self.map_size/2
            for i in range(self.n_agents):
                self.agent_pos.append([x[i],y])
                self.agent_yaw.append(np.pi/2)
                self.out_flag.append(0)
        else:
            raise Exception('No such agent mode')

        self._update_obs()

    def get_avail_agent_actions(self, agent_id):
        if agent_id >= self.n_agents:
            raise Exception('Agent id out of range')
        
        # avail_actions = np.zeros(self.n_actions)
        # yaw = self.agent_yaw[agent_id]
        # if abs(yaw) < np.pi/8:  
        #     avail_actions[:2] = 1
        #     avail_actions[-1] = 1
        # elif abs(yaw-np.pi/4) < np.pi/8:
        #     avail_actions[:3] = 1
        # elif abs(yaw-np.pi/2) < np.pi/8:
        #     avail_actions[1:4] = 1
        # elif abs(yaw-np.pi*3/4) < np.pi/8:
        #     avail_actions[2:5] = 1
        # elif abs(yaw-np.pi) < np.pi/8:
        #     avail_actions[3:6] = 1
        # elif abs(yaw-5*np.pi/4) < np.pi/8:
        #     avail_actions[4:7] = 1
        # elif abs(yaw-3*np.pi/2) < np.pi/8:
        #     avail_actions[5:] = 1
        # elif abs(yaw-7*np.pi/2) < np.pi/8:
        #     avail_actions[6:] = 1
        #     avail_actions[0] = 1
        # else:
        #     avail_actions[:] = 1
        # print(yaw, avail_actions)
        avail_actions = np.ones(self.n_actions)
        return avail_actions

    def get_state(self):
        # agent_state (x,y,yaw) shape:(n,3)
        pos = np.array(self.agent_pos)
        pos = (pos - 0.5*self.map_size)/(self.map_size/20)
        yaw = np.array(self.agent_yaw).reshape(-1,1)
        yaw = (yaw - np.pi)*3
        agent_state = np.hstack([pos, yaw])
        # print(pos.shape,yaw.shape, agent_state.shape)

        # target_state (x,y,find) shape:(m,3)
        pos = np.array(self.target_pos)   # (m,2)
        pos = (pos - 0.5*self.map_size)/(self.map_size/20)
        # priority = np.array([t.priority for t in self.target_list]).reshape(-1,1)     # (m,1)
        find = np.array([1 if t.find else 0 for t in self.target_list]).reshape(-1,1)    # (m,1)
        target_state = np.hstack([pos,find])
        state = np.hstack([agent_state.reshape(3*self.n_agents),target_state.reshape(3*self.target_num)])
        # print(pos.shape, find.shape, target_state.shape)
        # print(state.shape)
        return state    # shape (3*n+3*m)

    def get_obs(self):
        obs = np.array(self.obs)
        # print(obs.shape)
        # print('ori', obs)
        obs[:,:2] = (obs[:,:2]-0.5*self.map_size)/(self.map_size/20)
        obs[:,2] = (obs[:,2]-np.pi)*3
        # print('nor', obs)
        return obs

    def _update_obs(self):
        self.obs.clear()
        self.curr_reward = self.MOVE_COST
        for i in range(self.n_agents):
            x,y = self.agent_pos[i]
            yaw = self.agent_yaw[i]
            find = 0
            # out of the map punishment
            if self.out_flag[i] == 1:
                self.curr_reward += self.OUT_PUNISH

            # freqency reward
            self.freq_map[int(x), int(y)] += 1
            self.curr_reward += 1/self.freq_map[int(x), int(y)]

            # find target reward
            for target in self.target_list:
                t_x, t_y = target.pos
                if (t_x-x)**2 + (t_y-y)**2 <= self.view_range**2:
                    prob = np.random.rand()     # misdetect
                    if target.find == False and prob <= self.detect_prob:
                        find += 1
                        target.find = True
                        # self.curr_reward += target.priority*self.REWARD_FACTOR
                        self.curr_reward += self.REWARD_FACTOR
                        self.target_find += 1

            # collision punishment
            # for j in range(i+1, self.n_agents):
            #     x2,y2 = self.agent_pos[j]
            #     if (x-x2)**2 + (y-y2)**2 <= self.safe_dist**2:
            #         self.curr_reward += self.COLLI_PUNISH
            
            obs = [x,y,yaw,find]
            self.obs.append(obs)
    
    def _agent_step(self, act_list):
        if len(act_list) != self.n_agents:
            raise Exception('Act num mismatch agent')
        # dyaw = [0, np.pi/4, np.pi/2, np.pi*3/4, np.pi, np.pi*5/4, np.pi*3/2, np.pi*7/4]       
        dyaw = [-np.pi/18, 0, np.pi/18]  
        for i,(x,y) in enumerate(self.agent_pos):
            yaw = self.agent_yaw[i]
            yaw += dyaw[act_list[i]]    # change yaw of agent
            if yaw > 2*np.pi:     # yaw : 0~2pi
                yaw -= 2*np.pi
            elif yaw < 0:
                yaw += 2*np.pi
            x += self.velocity*np.cos(yaw)
            y += self.velocity*np.sin(yaw)

            # add potential-energy function to avoid collision
            f_x, f_y = self._potential_energy_force(i)
            x += f_x
            y += f_y
            # x = min(max(x, 0), self.map_size)
            # y = min(max(y, 0), self.map_size)
            if x < 0 or x > self.map_size or y < 0 or y > self.map_size:
                x = min(max(x, 0), self.map_size)
                y = min(max(y, 0), self.map_size)
                self.out_flag[i] = 1
            else:
                self.out_flag[i] = 0
            
            self.agent_pos[i] = [x,y]
            self.agent_yaw[i] = yaw
            # print(i, x, y, yaw)

    def _potential_energy_force(self, index):   # potential-energy force
        x, y = self.agent_pos[index]
        f_x, f_y = 0, 0
        for i, (x_a, y_a) in enumerate(self.agent_pos):
            if i != index and (x_a-x)**2+(y_a-y)**2 < self.force_dist**2:
                if x_a != x or y_a != y:
                    f_x += self.safe_dist*self.POTENTIAL_FORCE_FACTOR*self.velocity*(x-x_a)/((x-x_a)**2+(y-y_a)**2)
                    f_y += self.safe_dist*self.POTENTIAL_FORCE_FACTOR*self.velocity*(y-y_a)/((x-x_a)**2+(y-y_a)**2)
        return f_x, f_y

    def step(self, act_list):
        terminated = False
        info = ''
        
        self._agent_step(act_list)
        self._update_obs()
        self.total_reward += self.curr_reward
        self.time_step += 1

        if self.target_find >= self.target_num:
            terminated = True
        
        return self.curr_reward, terminated, info

    def render(self):
        plt.cla()
        COLORS = ['black', 'green', 'orange']
        for target in self.target_list:
            if target.find:
                plt.scatter(target.pos[0], target.pos[1], c=COLORS[2], s=7)
            else:
                plt.scatter(target.pos[0], target.pos[1], c=COLORS[0], s=7)
        for agent in self.agent_pos:
            [x,y] = agent
            plt.scatter(x, y, c='red', marker='^')
        title = 'target_find:{}/{}'.format(self.target_find, self.target_num)
        plt.title(title)
        plt.xlim(0, self.map_size)
        plt.ylim(0, self.map_size)
        plt.draw()
        if self.target_find == self.target_num:
            plt.pause(3)
        else:
            plt.pause(0.1)
    
    def close(self):
        pass



        