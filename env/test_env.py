import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

class Target():
    def __init__(self, pos, priority):
        self.pos = pos
        self.priority = priority
        self.find = False

class TestSearchEnv():
    def __init__(self):
        # env params 
        self.map_size = 100
        self.target_num = 1
        # self.agent_mode = 1
        self.n_agents = 1
        self.view_range = 10
        # self.circle_dict = circle_dict
        self.time_limit = 500
        self.turn_limit = np.pi/4
        self.detect_prob = 0.9
        self.safe_dist = 1
        self.velocity = 1
        self.force_dist = 3
        self.n_actions = 3
        self.state_shape = self.n_agents*4 + self.target_num*3 # agent: x,y,cos(yaw),sin(yaw) target: x,y,find
        self.obs_shape = 5 # x,y,cos(yaw),sin(yaw),find_num

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
        # self.last_dist = 0

        # reward
        # self.MOVE_COST = -0.01
        # self.REWARD_FACTOR = 20
        # # self.COLLI_PUNISH = -1
        # self.OUT_PUNISH = -0.01
        # self.FREQ_FACTOR = 0.01

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
            self.freq_map = np.zeros((self.map_size+1, self.map_size+1))
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
        x_t = 50
        y_t = 75
        target_tmp = Target([x_t,y_t], 1)
        self.target_list.append(target_tmp)
        self.target_pos.append([x_t,y_t])

        # reset agents
        x_a = 50
        y_a = 0
        self.agent_pos.append([x_a, y_a])
        self.agent_yaw.append(np.pi/2)

        self._update_obs()

    # def get_dist(self):
    #     dist = 0
    #     for i in range(self.n_agents):
    #         x, y = self.agent_pos[i]
    #         tx, ty = self.target_pos[0]
    #         dist += np.sqrt((x-tx)**2+(y-ty)**2)
    #     return dist

    def get_avail_agent_actions(self, agent_id):
        if agent_id >= self.n_agents:
            raise Exception('Agent id out of range')
        avail_actions = np.ones(self.n_actions)
        return avail_actions

    def get_state(self):
        # agent_state (x,y,cos,sin) shape:(n,4)
        obs = np.array(self.obs)
        agent_state = obs[:,:-1]
        agent_state[:,:2] = (agent_state[:,:2]-0.5*self.map_size)/(self.map_size/20)
        # print('agent_state:',agent_state.shape, '\n', agent_state)

        # target_state (x,y,find) shape:(m,3)
        pos = np.array(self.target_pos)   # (m,2)
        pos = (pos - 0.5*self.map_size)/(self.map_size/20)
        # priority = np.array([t.priority for t in self.target_list]).reshape(-1,1)     # (m,1)
        find = np.array([1 if t.find else 0 for t in self.target_list]).reshape(-1,1)    # (m,1)
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
        target_state[:,:2] = (target_state[:,:2]-0.5*self.map_size)/(self.map_size/20)
        # print('target_state:',target_state.shape, '\n', target_state)

        state = np.hstack([agent_state.reshape(4*self.n_agents),target_state.reshape(3*self.target_num)])
        # print('state:',state.shape, '\n', state)
        return state    # shape (4*n+3*m, )

    def get_obs(self):
        obs = np.array(self.obs)
        # print(obs.shape)
        # print('ori', obs)
        obs[:,:2] = (obs[:,:2]-0.5*self.map_size)/(self.map_size/20)
        # obs[:,2] = (obs[:,2]-np.pi)*3
        print('get obs ',obs.shape)
        return obs

    def _update_obs(self):
        self.obs.clear()
        # self.curr_reward = 0

        # dist reward
        # dist = self.get_dist()
        # self.curr_reward = self.last_dist - dist
        # print('last_dist:{} dist:{} reward:{}'.format(self.last_dist, dist, self.curr_reward))
        # self.last_dist = dist
        
        for i in range(self.n_agents):
            x,y = self.agent_pos[i]
            yaw = self.agent_yaw[i]
            find = 0
            # out of the map punishment
            # if self.out_flag[i] == 1:
            #     self.curr_reward += self.OUT_PUNISH

            # freqency reward
            # self.freq_map[int(x), int(y)] += 1
            # self.curr_reward += self.FREQ_FACTOR/self.freq_map[int(x), int(y)]

            # find target reward
            for target in self.target_list:
                t_x, t_y = target.pos
                if (t_x-x)**2 + (t_y-y)**2 <= self.view_range**2:
                    prob = np.random.rand()     # misdetect
                    if target.find == False and prob <= self.detect_prob:
                        find += 1
                        target.find = True
                        # self.curr_reward += target.priority*self.REWARD_FACTOR
                        # self.curr_reward += self.REWARD_FACTOR
                        self.target_find += 1

            # collision punishment
            # for j in range(i+1, self.n_agents):
            #     x2,y2 = self.agent_pos[j]
            #     if (x-x2)**2 + (y-y2)**2 <= self.safe_dist**2:
            #         self.curr_reward += self.COLLI_PUNISH
            
            obs = [x,y,np.cos(yaw),np.sin(yaw),find]
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
        win_info = False
        
        self._agent_step(act_list)
        self._update_obs()
        self.total_reward += self.curr_reward
        self.time_step += 1

        if self.target_find >= self.target_num:
            terminated = True
            win_info = True
        elif self.time_step >= self.time_limit:
            terminated = True

        return self.curr_reward, terminated, win_info

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


if __name__ == '__main__':
    t = TestSearchEnv()
    t.get_state()


        