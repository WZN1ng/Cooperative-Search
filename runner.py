import numpy as np
import os
from common.rollout import RolloutWorker
from agent.agent import Agents
from common.replay_buffer import ReplayBuffer
import matplotlib
# # matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Runner():
    def __init__(self, env, args):
        self.env = env
        if args.alg.find('qmix') > -1 or args.alg.find('dop') > -1 or args.alg.find('reinforce') > -1 \
            or args.alg.find('vdn') > -1 or args.alg.find('random') > -1:
            self.agents = Agents(env, args)
            self.rolloutWorker = RolloutWorker(env, self.agents, args)
        else:
            raise Exception('No such algorithm')

        if not args.show:
            if args.off_policy: # off-policy 
                self.buffer = ReplayBuffer(args, args.buffer_size)
        
        self.args = args
        self.win_rates = []
        self.targets_find = []
        self.episode_rewards = []

        self.result_path = args.result_dir + args.env + '_Seed' + str(args.seed) + '_' + args.alg + \
                            '_{}a{}t(AM{}TM{})'.format(args.n_agents, args.target_num, args.agent_mode, args.target_mode)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)  

        if args.alg != 'random':
            self.model_path = args.model_dir + args.env + '_Seed' + str(args.seed) + '_' + args.alg + \
                            '_{}a{}t(AM{}TM{})'.format(args.n_agents, args.target_num, args.agent_mode, args.target_mode)
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
        
    def run(self, num):
        train_steps = 0

        for epoch in range(self.args.n_epoch):
            print('Run {}, train epoch {}'.format(num, epoch))
            # if epoch % self.args.evaluate_cycle == 0 and epoch > 0:
            if epoch % self.args.evaluate_cycle == 0:
                win_rate, episode_reward, targets_find = self.evaluate()
                if self.args.search_env:
                    print('Average targets found :{}/{}'.format(targets_find, self.args.target_num))
                    print('Average episode reward :{}'.format(episode_reward))
                else:
                    print('Average episode reward :{}'.format(episode_reward))
                self.win_rates.append(win_rate)
                self.targets_find.append(targets_find)
                self.episode_rewards.append(episode_reward)
                self.plt(num)

            episodes = []
            # 收集self.args.n_episodes个episodes
            for episode_idx in range(self.args.n_episodes):
                episode, _, _, _ = self.rolloutWorker.generate_episode(episode_idx)
                episodes.append(episode)
                # print(_)
            # episode的每一项都是一个(1, episode_len, n_agents, 具体维度)四维数组，下面要把所有episode的的obs拼在一起
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)
            
            if not self.args.off_policy:
                self.agents.train(episode_batch, train_steps, self.rolloutWorker.epsilon)
                train_steps += 1
            else:
                self.buffer.store_episode(episode_batch)
                for train_step in range(self.args.train_steps):
                    mini_batch = self.buffer.sample(min(self.buffer.current_size, self.args.batch_size))
                    if self.args.alg == 'dop':
                        self.agents.train(mini_batch, train_steps, self.rolloutWorker.epsilon)
                    else:
                        self.agents.train(mini_batch, train_steps)
                    train_steps += 1
        self.plt(num)
    
    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        cumu_targets = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, targets_find = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            cumu_targets += targets_find
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch, cumu_targets / self.args.evaluate_epoch

    def plt(self, num):
        plt.figure()
        plt.axis([0, self.args.n_epoch, 0, 100])
        plt.cla()
        if self.args.search_env:
            plt.subplot(2, 1, 1)
            plt.plot(range(len(self.targets_find)), self.targets_find)
            plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
            plt.ylabel('targets_find')

            plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('epoch*{}'.format(self.args.evaluate_cycle))
        plt.ylabel('episode_rewards')

        plt.savefig(os.path.join(self.result_path, 'plt_{}.png'.format(num)), format='png')
        if self.args.search_env:
            np.save(os.path.join(self.result_path, 'targets_find_{}'.format(num)), self.targets_find)
        np.save(os.path.join(self.result_path, 'episode_rewards_{}'.format(num)), self.episode_rewards)

    def replay(self, num):  # the num of model loaded

        if self.args.alg == 'qmix':
            rnn_root = os.path.join(self.model_path, str(num) + '_rnn_net_params.pkl')
            qmix_root = os.path.join(self.model_path, str(num) + '_qmix_net_params.pkl')
            self.agents.policy.load_model(rnn_root, qmix_root)
        elif self.args.alg == 'reinforce':
            rnn_root = os.path.join(self.model_path, str(num) + '_rnn_net_params.pkl')
            self.agents.policy.load_model(rnn_root)
        elif self.args.alg == 'dop':
            actor_root = os.path.join(self.model_path, str(num) + '_actor_net_params.pkl')
            critic_root = os.path.join(self.model_path, str(num) + '_critic_net_params.pkl')
            mixer_root = os.path.join(self.model_path, str(num) + '_mixer_net_params.pkl')
            self.agents.policy.load_model(actor_root, critic_root, mixer_root)
            
        else:
            raise Exception('Unknown Algorithm model to load')
        
        targets_find, episode_reward, res, step = self.rolloutWorker.generate_replay(render=True)
        print('targets_find: ', targets_find, ' reward: ', episode_reward)

    def collect_experiment_data(self, num, replay_times):
        tgt_find_list, reward_list, res_list, step_list = [], [], [], []
        for i in range(replay_times):
            # load model
            if self.args.alg == 'qmix':
                rnn_root = os.path.join(self.model_path, str(num) + '_rnn_net_params.pkl')
                qmix_root = os.path.join(self.model_path, str(num) + '_qmix_net_params.pkl')
                self.agents.policy.load_model(rnn_root, qmix_root)
            elif self.args.alg == 'dop':
                actor_root = os.path.join(self.model_path, str(num) + '_actor_net_params.pkl')
                critic_root = os.path.join(self.model_path, str(num) + '_critic_net_params.pkl')
                mixer_root = os.path.join(self.model_path, str(num) + '_mixer_net_params.pkl')
                self.agents.policy.load_model(actor_root, critic_root, mixer_root)
            elif self.args.alg == 'reinforce':
                actor_root = os.path.join(self.model_path, str(num) + '_rnn_net_params.pkl')
                self.agents.policy.load_model(actor_root)

            targets_find, episode_reward, res, step = self.rolloutWorker.generate_replay(collect=True, render=False)
            tgt_find_list.append(targets_find)
            reward_list.append(episode_reward)
            res_list.append(res)
            step_list.append(step)
            # print('experiment {} finished'.format(i))
        
        average_tgt_find = np.mean(tgt_find_list)
        average_rew = np.mean(reward_list)
        average_step = np.mean(step_list)
        res_list = np.array(res_list)
        average_res = np.mean(res_list, axis=0)
        idx_list = [10, 20, 40, 60, 80, 100, 150, 199]
        print(average_tgt_find, average_rew, average_step)
        print(average_res[idx_list]*100)
        np.save(os.path.join(self.result_path, 'average_res_{}'.format(num)), average_res*100)
        print('process data saved!')
