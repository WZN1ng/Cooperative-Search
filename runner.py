import numpy as np
import os
from common.rollout import RolloutWorker
from agent.agent import Agents
from common.replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt


class Runner():
    def __init__(self, env, args):
        self.env = env

        if args.alg.find('qmix') > -1:
            self.agents = Agents(args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        
        if args.learn and args.alg.find('qmix') > -1:
            self.buffer = ReplayBuffer(args)
        self.args = args
        self.win_rates = []
        self.episode_rewards = []

        self.save_path = self.args.result_dir + '/' + args.alg + '/{}X{}_{}agents_{}targets'.format(
                                                args.map_size, args.map_size, args.n_agents, args.target_num)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
    def run(self, num):
        train_steps = 0
        # print('Run {} start'.format(num))
        for epoch in range(self.args.n_epoch):
            print('Run {}, train epoch {}'.format(num, epoch))
            if epoch % self.args.evaluate_cycle == 0:
                win_rate, episode_reward = self.evaluate()
                print('win_rate is ', win_rate)
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                self.plt(num)

            episodes = []
            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):
                episode, _, _ = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                # print(_)
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            self.buffer.store_episode(episode_batch)
            for train_step in range(self.args.train_steps):
                mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                self.agents.train(mini_batch, train_steps)
                train_steps += 1
        self.plt(num)
    
    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def plt(self, num):
        plt.figure()
        plt.axis([0, self.args.n_epoch, 0, 100])
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('win_rate')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        plt.savefig(self.save_path + '/plt_{}.png'.format(num), format='png')
        np.save(self.save_path + '/win_rates_{}'.format(num), self.win_rates)
        np.save(self.save_path + '/episode_rewards_{}'.format(num), self.episode_rewards)

    def replay(self, num):  # the num of model loaded
        if self.args.alg == 'qmix':
            model_root = self.args.model_dir + '{}/{}X{}_{}agents_{}targets/'.format(self.args.alg, 
                                        str(self.args.map_size), str(self.args.map_size), str(self.args.n_agents), str(self.args.target_num)) 
            rnn_root = model_root + str(num) + '_rnn_net_params.pkl'
            qmix_root = model_root + str(num) + '_qmix_net_params.pkl'
            self.agents.policy.load_model(rnn_root, qmix_root)
            self.rolloutWorker.generate_replay()
        else:
            raise Exception('Unknown Algorithm model to load')

