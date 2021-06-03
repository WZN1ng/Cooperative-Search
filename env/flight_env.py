import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy import signal
import cv2
import os

class Target():
    def __init__(self, pos, priority):
        self.pos = pos
        self.priority = priority
        self.find = False

class FlightSearchEnv():
    def __init__(self, args, circle_dict):
        # env params 
        self.args = args
        self.map_size = args.map_size
        self.target_num = args.target_num
        self.target_mode = args.target_mode
        self.agent_mode = args.agent_mode
        self.n_agents = args.n_agents
        self.view_range = args.view_range
        self.circle_dict = circle_dict
        self.time_limit = args.time_limit
        self.turn_limit = args.turn_limit
        self.detect_prob = args.detect_prob
        self.wrong_alarm_prob = args.wrong_alarm_prob
        self.safe_dist = args.safe_dist
        self.velocity = args.agent_velocity
        self.force_dist = args.force_dist
        self.n_actions = 3
        self.state_shape = self.n_agents*4 + self.target_num*3 # agent: x,y,cos(yaw),sin(yaw) target: x,y,find
        self.obs_shape = 4
        # self.obs_shape = self.map_size**2 + 5 # x,y,cos(yaw),sin(yaw),find_num + freq_map
        # self.prob_map_root = args.model_dir + args.env + '_Seed' + str(args.seed) + '_' + args.alg + '_{}a{}t'.format(args.n_agents, args.target_num)
        print('Init Env ' + args.env + ' {}a{}t(agent mode:{}, target mode:{})'.format(self.n_agents, self.target_num, self.agent_mode, self.target_mode))

        # env variables
        self.time_step = 0
        self.target_list = []
        self.target_pos = []
        self.target_find = 0
        # self.wrong_alarm = 0
        self.agent_pos = []
        self.agent_yaw = []
        self.total_reward = 0
        self.curr_reward = 0
        self.find = []
        self.obs = []
        self.win_flag = False
        self.out_flag = []          # flag = 1 only when agent get out of the map
        self.prob_map = 0.5*np.ones((self.map_size, self.map_size))  # reused probability map
        # self.freq_map = np.ones((self.map_size, self.map_size))
        self.target_map = dict()

        # reward
        self.FIND_ONE_TGT = 10
        self.FIND_ALL_TGT = 100
        # self.TIME_STEP_1 = self.time_limit//4
        # self.TIME_STEP_2 = 3*self.TIME_STEP_1
        # self.TIME_FACTOR = 0.1
        # self.FREQ_FACTOR = 2.0
        # self.FREQ_REW_FACTOR = 1.0 
        self.OUT_PUNISH = -1
        self.MOVE_COST = -1

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
        if init:
            # self.freq_map = np.ones((self.map_size, self.map_size))
            self.prob_map = 0.5*np.ones((self.map_size, self.map_size))
        self.target_map = dict()
        self.time_step = 0
        self.target_find = 0
        # self.wrong_alarm = 0
        self.total_reward = 0
        self.curr_reward = 0
        self.obs.clear()
        self.target_list.clear()
        self.target_pos.clear()
        self.agent_pos.clear()
        self.agent_yaw.clear()
        self.out_flag.clear()
        self.win_flag = False
        self.find.clear()

        # reset targets 
        # reset targets 
        if self.target_mode == 0:   # load targets from file
            for i in range(self.target_num):
                a = self.map_size/10
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

                # target map
                x_idx, y_idx = min(int(x), self.map_size - 1), min(int(y), self.map_size - 1)
                if (x_idx, y_idx) not in self.target_map:
                    self.target_map[(x_idx, y_idx)] = [i]
                else:
                    self.target_map[(x_idx, y_idx)].append(i)

        elif self.target_mode == 1:  # totally random
            for i in range(self.target_num):
                x, y = [self.map_size*np.random.rand() for _ in range(2)]
                target_tmp = Target([x,y], 1)
                self.target_list.append(target_tmp)
                self.target_pos.append([x,y])

                # target map
                x_idx, y_idx = min(int(x), self.map_size-1), min(int(y), self.map_size-1)
                if (x_idx, y_idx) not in self.target_map:
                     self.target_map[(x_idx, y_idx)] = [i]
                else:
                    self.target_map[(x_idx, y_idx)].append(i)
        else:
            raise Exception('No such target mode')

        # reset agents
        if self.agent_mode == 0: # start from the bottom line(left corner, midium, right corner)
            if self.n_agents != 1:
                x = [i*self.map_size/(self.n_agents-1) for i in range(self.n_agents)]
            else:
                x = [self.map_size/2]
            y = 0
            for i in range(self.n_agents):
                self.agent_pos.append([x[i], y])
                self.agent_yaw.append(np.pi/2)
                self.out_flag.append(0)
        elif self.agent_mode == 1:  # start from the midium of the map
            if self.n_agents != 1:
                x = [i*self.map_size/(self.n_agents-1) for i in range(self.n_agents)]
            else:
                x = [self.map_size/2]
            y = self.map_size/2
            for i in range(self.n_agents):
                self.agent_pos.append([x[i], y])
                self.agent_yaw.append(np.pi/2)
                self.out_flag.append(0)
        elif self.agent_mode == 2:  # start from the left line
            if self.n_agents != 1:
                y = [i*self.map_size/(self.n_agents-1) for i in range(self.n_agents)]
            else:
                y = [self.map_size/2]
            x = 0 
            for i in range(self.n_agents):
                self.agent_pos.append([x, y[i]])
                self.agent_yaw.append(0)
                self.out_flag.append(0)
        elif self.agent_mode == 3:  # start from the right line
            if self.n_agents != 1:
                y = [i*self.map_size/(self.n_agents-1) for i in range(self.n_agents)]
            else:
                y = [self.map_size/2]
            x = self.map_size
            for i in range(self.n_agents):
                self.agent_pos.append([x, y[i]])
                self.agent_yaw.append(np.pi)
                self.out_flag.append(0)
        else:
            raise Exception('No such agent mode')

        self._update_obs()

    def get_avail_agent_actions(self, agent_id):
        if agent_id >= self.n_agents:
            raise Exception('Agent id out of range')
        avail_actions = np.ones(self.n_actions)
        return avail_actions

    def get_state(self):
        # agent_state (x,y,cos,sin) shape:(n,4)
        agent_state = np.array(self.obs)
        agent_state[:, :2] = (agent_state[:, :2]-0.5*self.map_size)/(self.map_size/2)
        # print('agent_state:',agent_state.shape, '\n', agent_state)

        # target_state (x,y,find) shape:(m,3)
        target_state = []
        for i in range(self.target_num):
            s = []
            s.extend(self.target_pos[i])
            if self.target_list[i].find:
                s.append(1.0)
            else:
                s.append(0.0)
            target_state.append(s)
        target_state = np.array(target_state)
        target_state[:,:2] = (target_state[:,:2]-0.5*self.map_size)/(self.map_size/2)
        # print('target_state:',target_state.shape, '\n', target_state)

        state = np.hstack([agent_state.reshape(4*self.n_agents), target_state.reshape(3*self.target_num)])
        # print('state:',state.shape, '\n', state)
        return state    # shape (4*n+3*m, )   # shape (4*n+3*m, )

    def get_obs(self):
        obs = np.array(self.obs)
        obs[:,:2] = (obs[:,:2] - 0.5*self.map_size)/(self.map_size/2)
        prob = self.prob_map.reshape(1, self.map_size*self.map_size)
        prob = np.repeat(prob, self.n_agents, axis=0)
        obs = np.concatenate((prob, obs), axis=1)
        # print('obs: ', obs.shape)
        return obs

    def _update_obs(self):
        self.obs.clear()
        self.curr_reward = 0

        # move cost
        self.curr_reward += self.MOVE_COST
        tgt_found = []

        for i in range(self.n_agents):
            x, y = self.agent_pos[i]
            yaw = self.agent_yaw[i]

            # find target reward
            for j, target in enumerate(self.target_list):
                t_x, t_y = target.pos
                if (t_x-x)**2 + (t_y-y)**2 <= self.view_range**2:
                    prob = np.random.rand()     # misdetect
                    if target.find == False and prob <= self.detect_prob:
                        target.find = True
                        self.curr_reward += self.FIND_ONE_TGT
                        self.target_find += 1
                        tgt_found.append(j)

                        if self.target_find == self.target_num and self.win_flag == False:
                            self.curr_reward += self.FIND_ALL_TGT
                            self.win_flag = True

            # out of map punishment
            if self.out_flag[i]:
                self.curr_reward += self.OUT_PUNISH
            
            obs = np.array([x,y,np.cos(yaw),np.sin(yaw)])
            self.obs.append(obs)
        
        self._update_prob_map(tgt_found)

    # def _cal_eta(self, prob_map):
    #     t_sum = 0
    #     for i in range(self.map_size):
    #         for j in range(self.map_size):
    #             t_sum += abs(prob_map[i,j])
    #     return np.exp(-self.FREQ_FACTOR*t_sum)

    def _update_prob_map(self, tdx_list):
        tgt_pos_list = [self.target_pos[x]  for x in tdx_list]
        tgt_idx_list = []
        for tgt_pos in tgt_pos_list:
            idx, idy = min(int(tgt_pos[0]), self.map_size-1), min(int(tgt_pos[1]), self.map_size-1)
            tgt_idx_list.append([idx, idy])

        for i in range(self.map_size):
            for j in range(self.map_size):
                percent = self._percent_in_agent_viewrange(i, j)
                if percent == 0:
                    continue
                else:
                    if [i,j] in tgt_idx_list:
                        self.prob_map[i,j] = 1
                    else:
                        p = self.prob_map[i,j]
                        self.prob_map[i,j] = percent*(1-self.detect_prob)*p/((1-self.detect_prob)*p + (1-p))

    def _percent_in_agent_viewrange(self, i, j):
        cell_length = 1
        corner_points = [[i, j], [i+cell_length, j], [i, j+cell_length], [i+cell_length, j+cell_length]]
        in_point = 0
        for x, y in corner_points:
            for ax, ay in self.agent_pos:
                if (x-ax)**2 + (y-ay)**2 < self.view_range**2:
                    in_point += 1
                    break
        return in_point / 4

    def _agent_step(self, act_list):
        if len(act_list) != self.n_agents:
            raise Exception('Act num mismatch agent')
        # dyaw = [0, np.pi/4, np.pi/2, np.pi*3/4, np.pi, np.pi*5/4, np.pi*3/2, np.pi*7/4]       
        dyaw = [0, np.pi/18, -np.pi/18]
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

            # check whether go out of the map
            if x < 0 or x >= self.map_size or y < 0 or y >= self.map_size:
                x = min(max(x, 0), self.map_size)
                y = min(max(y, 0), self.map_size)
                if yaw <= np.pi:
                    yaw = np.pi - yaw 
                else:
                    yaw = 3*np.pi - yaw
                self.out_flag[i] = 1
            else:
                self.out_flag[i] = 0

            # update freq map
            idx, idy = min(self.map_size-1, int(x)), min(self.map_size-1, int(y))
            # self.freq_map[idx, idy] += 1
            
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

        self._agent_step(act_list)
        self._update_obs()
        self.total_reward += self.curr_reward
        self.time_step += 1

        if self.target_find >= self.target_num or self.time_step >= self.time_limit:
            terminated = True
        
        return self.curr_reward, terminated, self.win_flag

    # def save_prob_map(self, num):
    #     idx = str(num)
    #     np.save(os.path.join(self.prob_map_root, idx + '_prob_map'), self.prob_map)

    # def load_prob_map(self, num):
    #     idx = str(num)
    #     self.prob_map = np.load(os.path.join(self.prob_map_root, idx + '_prob_map'))

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



        