from policy.qmix import QMIX
from policy.dop import DOP
from policy.reinforce import Reinforce
from policy.vdn import VDN
from policy.trandition import Random
from torch.distributions import Categorical
import numpy as np
import torch

class Agents():
    def __init__(self, env, args):
        self.map_size = args.map_size
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.env = env
        if args.alg == 'qmix':
            self.policy = QMIX(args)
        elif args.alg == 'dop':
            self.policy = DOP(args)
        elif args.alg == 'reinforce':
            self.policy = Reinforce(args)
        elif args.alg == 'vdn':
            self.policy = VDN(args)
        elif args.alg == 'random':
            self.policy = Random(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        print('Init Agents')
    
    def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, evaluate=False):
        if self.args.alg == 'random':
            avail_actions_ind = np.nonzero(avail_actions)[0] 
            action = np.random.choice(avail_actions_ind)
        else:
            inputs = obs.copy()
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
            if self.args.alg == 'dop':
                q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.actor(inputs, hidden_state)
            else:
                q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(inputs, hidden_state)

            # choose action from q value
            if self.args.alg == 'reinforce':
                action = self._choose_action_from_softmax(q_value.cpu(), avail_actions, epsilon, evaluate)
            else:
                q_value[avail_actions == 0.0] = - float("inf")
                if evaluate or np.random.rand() >= epsilon:
                    action = torch.argmax(q_value)
                else:
                    action = np.random.choice(avail_actions_ind)
        return action
    
    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon, evaluate=False):
        """
        :param inputs: # q_value of all actions
        """
        action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # num of avail_actions
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        # add noise of epsilon
        prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """

        if epsilon == 0 and evaluate:
            action = torch.argmax(prob)
        else:
            action = Categorical(prob).sample().long()
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
        # if on_batch:
        #     max_episode_len = max(self._get_max_episode_len(batch), self._get_max_episode_len(on_batch))
        # else:
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            # print(batch[key].shape)
            # if on_batch:
            #     on_batch[key] = on_batch[key][:, :max_episode_len]
            batch[key] = batch[key][:, :max_episode_len]

        # if self.args.alg == 'dop':
            # if on_batch:
            #     self.policy.train_critic(on_batch, max_episode_len, train_step, epsilon, best_batch=batch)
            # else:
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        # else:
        #     self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            idx = self.policy.get_model_idx()
            self.policy.save_model(idx)
            # if self.args.env == 'flight':
            #     self.env.save_prob_map(idx)