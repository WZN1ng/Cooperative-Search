import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time

class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        print('Init RolloutWorker')

    def generate_episode(self, episode_num=None, evaluate=False):
        if evaluate and episode_num == 0:  # prepare for save replay of evaluation
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset()
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        # if self.args.alg == 'qmix':
        self.agents.policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == 'episode':
            epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        if self.args.epsilon_anneal_scale == 'epoch':
            if episode_num == 0:
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon

        while terminated == False and step < self.episode_limit:
            # time.sleep(0.2)
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []
            # if step % 100 == 0:
            #     print(evaluate, step, epsilon)
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                # print(obs.shape, last_action.shape)
                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id, \
                                                       avail_action, epsilon, evaluate)

                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot
            reward, terminated, info = self.env.step(actions)
            win_tag = True if terminated and info else False
            o.append(obs)
            s.append(state)
            u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            r.append([reward])
            terminate.append([terminated])
            padded.append([0.])
            episode_reward += reward
            step += 1
            if self.args.epsilon_anneal_scale == 'step':
                epsilon = epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
        
        if self.args.search_env:
            targets_find = self.env.target_find     # 

        # last obs
        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        if self.args.conv:
            obs_shape = self.obs_shape + self.args.map_size**2
        else:
            obs_shape = self.obs_shape

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, obs_shape)))
            u.append(np.zeros([self.n_agents, 1]))
            s.append(np.zeros(self.state_shape))
            r.append([0.])
            o_next.append(np.zeros((self.n_agents, obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            padded.append([1.])
            terminate.append([1.])

        episode = dict(o=o.copy(),
                       s=s.copy(),
                       u=u.copy(),
                       r=r.copy(),
                       avail_u=avail_u.copy(),
                       o_next=o_next.copy(),
                       s_next=s_next.copy(),
                       avail_u_next=avail_u_next.copy(),
                       u_onehot=u_onehot.copy(),
                       padded=padded.copy(),
                       terminated=terminate.copy()
                       )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])
        if not evaluate:
            self.epsilon = epsilon
        if evaluate and episode_num == self.args.evaluate_epoch - 1:
            self.env.close()
        if self.args.search_env:
            return episode, episode_reward, win_tag, targets_find
        else:
            return episode, episode_reward, False, False

    # show the model
    def generate_replay(self, force=0, render=True, collect=False):
        # o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []
        self.env.reset(init=True)
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative rewards
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        # if self.args.alg == 'qmix':
        self.agents.policy.init_hidden(1)

        epsilon = 0
        evaluate = True
        res = []

        while not terminated and step < self.episode_limit:
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                action = self.agents.choose_action(obs[agent_id], last_action[agent_id], agent_id,
                                                       avail_action, epsilon, evaluate)
                # generate onehot vector of th action
                action_onehot = np.zeros(self.args.n_actions)
                action_onehot[action] = 1
                actions.append(action)
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot
            # act_list = [a.detach().cpu().numpy().reshape(-1)[0] for a in actions]
            reward, terminated, info = self.env.step(actions)

            if render:
                self.env.render()
            win_tag = True if terminated else False
            # o.append(obs)
            # s.append(state)
            # u.append(np.reshape(actions, [self.n_agents, 1]))
            # u_onehot.append(actions_onehot)
            # avail_u.append(avail_actions)
            # r.append([reward])
            # terminate.append([terminated])
            # padded.append([0.])
            episode_reward += reward
            step += 1

            tgt_find = self.env.target_find
            res.append(tgt_find/self.args.target_num)
        
        # if not collect:
        #     # for i in range(len(res)):
        #         # print('{} steps, agents found {:.2f}% targets'.format(time_step_standard[i], res[i]*100))
        #     print('{} steps, agents found all targets'.format(step))
        for _ in range(self.episode_limit-len(res)):
            res.append(1.0)

        if self.args.search_env:
            targets_find = self.env.target_find
            return targets_find, episode_reward, res, step
        else:
            return 0, episode_reward, res, step
            

    