from policy.qmix import QMIX
from policy.dop import DOP

import numpy as np
import torch

class Agents():
    def __init__(self, args):
        self.map_size = args.map_size
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        if args.alg == 'qmix':
            self.policy = QMIX(args)
        elif args.alg == 'dop':
            self.policy = DOP(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        print('Init Agents')
    
    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate=False):
        inputs = obs.copy()
        # x,y --> [-10,10]
        # inputs[:2] = (inputs[:2]-0.5*self.map_size)/(self.map_size/20)
        # yaw --> [-3pi,3pi]
        # inputs[2] = (inputs[2]-np.pi)*3
        # print(inputs)
        avail_actions_ind = np.nonzero(avail_actions)[0]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        # if self.args.alg == 'qmix':
        hidden_state = self.policy.eval_hidden[:, agent_num, :]

        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()  
            # if self.args.alg == 'qmix':
            hidden_state = hidden_state.cuda()

        # get q value
        if self.args.alg == 'qmix':
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)
        elif self.args.alg == 'dop':
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_actor(inputs, hidden_state)

        # choose action from q value
        q_value[avail_actions == 0.0] = - float("inf")
        if np.random.uniform() < epsilon:
            action = np.random.choice(avail_actions_ind)  # action是一个整数
        else:
            action = torch.argmax(q_value)
        return action
    
    def _get_max_episode_len(self, batch):
        terminated = batch['terminated']
        episode_num = terminated.shape[0]
        max_episode_len = batch['o'].shape[1]
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        # print('1 ',max_episode_len)
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training

        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            # print(batch[key].shape)
            batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)