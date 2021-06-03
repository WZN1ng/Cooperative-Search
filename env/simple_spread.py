import numpy as np
import matplotlib.pyplot as plt
import os

class Target():
    def __init__(self, pos, radius):
        self.pos = pos
        self.radius = radius

class Agent():
    def __init__(self, pos, radius):
        self.pos = pos
        self.radius = radius

class SimpleSpreadEnv():
    def __init__(self, args):
        # env params
        self.args = args
        self.map_size = args.map_size
        self.target_num = args.target_num
        self.target_radius = 1
        self.n_agents = args.n_agents
        self.agent_radius = 6
        self.time_limit = 100
        self.n_actions = 5
        self.state_shape = self.n_agents*2 + self.target_num*2
        self.obs_shape = 2 + (self.n_agents-1)*2 + self.target_num*4
        print('Init Env ' + args.env + ' {}a{}t'.format(self.n_agents, self.target_num))

        # env variables 
        self.time_step = 0
        self.target_list = []
        self.agent_list = []
        self.total_reward = 0
        self.curr_reward = 0
        self.obs = []
        self.occupied = []
        self.colli = np.zeros(self.n_agents)

        self._update_obs()
    
    def get_env_info(self):
        env_info = {'n_actions': self.n_actions,
                    'state_shape': self.state_shape,
                    'obs_shape': self.obs_shape,
                    'episode_limit': self.time_limit}
        return env_info

    def reset(self, init=False):
        self.time_step = 0
        self.total_reward = 0
        self.curr_reward = 0
        self.obs.clear()
        self.target_list.clear()
        self.agent_list.clear()
        self.occupied.clear()

        # reset targets
        for i in range(self.target_num):
            x, y = [self.map_size*np.random.rand() for _ in range(2)]
            tgt_tmp = Target([x,y], self.target_radius)
            self.target_list.append(tgt_tmp)
            self.occupied.append(0)
        
        # reset agents
        for i in range(self.n_agents):
            x, y = [self.map_size*np.random.rand() for _ in range(2)]
            agent_tmp = Agent([x,y], self.agent_radius)
            self.agent_list.append(agent_tmp)

        self._update_obs()

    def get_avail_agent_actions(self, agent_id):
        if agent_id >= self.n_agents:
            raise Exception('Agent id out of range')
        avail_actions = np.ones(self.n_actions)
        return avail_actions
    
    def _update_obs(self):
        self.obs.clear()

        # update obs
        for i, agent in enumerate(self.agent_list):
            obs_tmp = []
            obs_tmp.append((agent.pos[0]-0.5*self.map_size))
            obs_tmp.append((agent.pos[1]-0.5*self.map_size))
            for j, tgt in enumerate(self.target_list):
                for k in range(2):
                    obs_tmp.append(tgt.pos[k]-agent.pos[k])
            for j, other in enumerate(self.agent_list):
                if i != j:
                    for k in range(2):
                        obs_tmp.append(other.pos[k]-agent.pos[k])
            self.obs.append(obs_tmp)

        # occupied
        for i, tgt in enumerate(self.target_list):
            occupied = 0
            for j in range(self.n_agents):
                self.obs[j].append((tgt.pos[0]-0.5*self.map_size))
                self.obs[j].append((tgt.pos[1]-0.5*self.map_size))
            for j, agent in enumerate(self.agent_list):
                dis = 0 
                for k in range(2):
                    dis += (tgt.pos[k]-agent.pos[k])**2
                dis = np.sqrt(dis)
                if dis < self.agent_radius:
                    occupied = 1
                    break
            self.occupied[i] = occupied

    def get_obs(self):
        # print(np.array(self.obs).shape)
        return np.array(self.obs)

    def get_state(self):
        state = []
        for i, agent in enumerate(self.agent_list):
            for k in range(2):
                state.append((agent.pos[k]-0.5*self.map_size))
        for i, tgt in enumerate(self.target_list):
            for k in range(2):
                state.append((tgt.pos[k]-0.5*self.map_size))
        
        # state.append(sum(self.occupied)/2)
        # state.append(sum(self.colli)/2)
        # print(state)
        return np.array(state)
        
    def _agent_step(self, act_list):
        if len(act_list) != self.n_agents:
            raise Exception('Act num mismatch agent')

        dpos = [[0,0],[1,0],[0,1],[-1,0],[0,-1]]
        for i, agent in enumerate(self.agent_list):
            for k in range(2):
                agent.pos[k] += dpos[act_list[i]][k]
                agent.pos[k] = min(max(0, agent.pos[k]), self.map_size)

    def reward(self):
        reward = 0
        self.colli = np.zeros(self.n_agents)

        for j,tgt in enumerate(self.target_list):
            dis = []
            for i,agent in enumerate(self.agent_list):
                dis_tmp = 0
                for k in range(2):
                    dis_tmp += (agent.pos[k]-tgt.pos[k])**2
                dis_tmp = np.sqrt(dis_tmp)
                dis.append(dis_tmp)
            reward -= min(dis)

        # for i, agent in enumerate(self.agent_list):
        #     for j in range(i+1, self.n_agents):
        #         other = self.agent_list[j]
        #         dis_tmp = 0
        #         for k in range(2):
        #             dis_tmp += (agent.pos[k]-other.pos[k])**2
        #         dis_tmp = np.sqrt(dis_tmp)
        #         if dis_tmp < 2*self.agent_radius:
        #             reward -= 2
        #             self.colli[i] = 1
        #             self.colli[j] = 1
        return reward
        
    def step(self, act_list):
        terminated = False

        self._agent_step(act_list)
        self._update_obs()
        self.curr_reward = self.reward()
        self.total_reward += self.curr_reward
        self.time_step += 1

        if self.time_step >= self.time_limit:
            terminated = True
        return self.curr_reward, terminated, False

    def render(self):
        plt.cla()
        alpha = 400
        for i,tgt in enumerate(self.target_list):
            if self.occupied[i]:
                plt.scatter(tgt.pos[0], tgt.pos[1], s=alpha*self.target_radius, c='black', alpha=0.2)
            else:
                plt.scatter(tgt.pos[0], tgt.pos[1], s=alpha*self.target_radius, c='black', alpha=1)
        for i,agent in enumerate(self.agent_list):
            if self.colli[i]:
                plt.scatter(agent.pos[0], agent.pos[1], s=alpha*self.agent_radius, c='red', alpha=0.5)
            else:
                plt.scatter(agent.pos[0], agent.pos[1], s=alpha*self.agent_radius, c='blue', alpha=0.5)
        title = "Reward: {}".format(self.total_reward)
        plt.title(title)
        plt.xlim(0, self.map_size)
        plt.ylim(0, self.map_size)
        plt.draw()
        plt.pause(0.1)

    def close(self):
        pass


        