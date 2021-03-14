from network.offpg_net import OffPGCritic
from network.mixer_net import MixerNet
from network.base_net import RNN

import torch
import os
import numpy as np

class DOP():
    def __init__(self, args):
        # params
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape

        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents
        critic_input_shape = input_shape + self.n_actions 
        
        # nets
        self.eval_actor = RNN(input_shape, args)
        self.target_actor = RNN(input_shape, args)
        self.eval_critic = OffPGCritic(critic_input_shape, args)
        self.target_critic = OffPGCritic(critic_input_shape, args)
        self.eval_mixer_net = MixerNet(args)
        self.target_mixer_net = MixerNet(args)
        if self.args.cuda:
            self.eval_actor.cuda()
            self.target_actor.cuda()
            self.eval_critic.cuda()
            self.target_critic.cuda()
            self.eval_mixer_net.cuda()
            self.target_mixer_net.cuda()
        self.model_dir = args.model_dir + args.alg + '/{}X{}_{}agents_{}targets/'.format(
                                                args.map_size, args.map_size, self.n_agents, args.target_num)

        # load model
        if self.args.load_model:
            if os.path.exists(self.model_dir + str(args.model_index) + '_critic_net_params.pkl'):
                path_actor = self.model_dir + str(args.model_index) + '_actor_net_params.pkl'
                path_mixer = self.model_dir + str(args.model_index) + '_mixer_net_params.pkl'
                path_critic = self.model_dir + str(args.model_index) + '_critic_net_params.pkl'
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_actor.load_state_dict(torch.load(path_actor, map_location=map_location))
                self.eval_critic.load_state_dict(torch.load(path_critic, map_location=map_location))
                self.eval_mixer_net.load_state_dict(torch.load(path_mixer, map_location=map_location))
                print('Successfully load the model: {}, {} and {}'.format(path_actor, path_critic, path_mixer))
            else:
                raise Exception('No such model')

        # update target net params
        self.target_actor.load_state_dict(self.eval_actor.state_dict())
        self.target_critic.load_state_dict(self.eval_critic.state_dict())
        self.target_mixer_net.load_state_dict(self.eval_mixer_net.state_dict())

        self.actor_params = list(self.eval_actor.parameters())
        self.critic_params = list(self.eval_critic.parameters())
        self.mixer_params = list(self.eval_mixer_net.parameters())
        self.params = self.actor_params + self.critic_params
        self.c_params = self.critic_params + self.mixer_params 

        if args.optimizer == 'RMS':
            self.agent_optimizer = torch.optim.RMSprop(self.actor_params, lr=args.lr)
            self.critic_optimizer = torch.optim.RMSprop(self.critic_params, lr=args.critic_lr)
            self.mixer_optimizer = torch.optim.RMSprop(self.mixer_params, lr=args.critic_lr)
        else:
            raise Exception('No such optimizer')

        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg DOP')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):
        print('learn')
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                             batch['r'],  batch['avail_u'], batch['avail_u_next'],\
                                                             batch['terminated']
        mask = 1 - batch['padded'].float()
        
        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
        
        q_vals, _ = self.get_q_output(batch, max_episode_len)
        actor_out = self.get_actor_output(batch, max_episode_len)

        actor_out[avail_u == 0] = 0
        actor_out = actor_out/actor_out.sum(dim=1, keepdim=True)
        actor_out[avail_u == 0] = 0

        q_taken = torch.gather(q_vals, dim=3, index=u).squeeze(3)
        pi = actor_out.view(-1, self.n_actions)
        baseline = torch.sum(actor_out*q_vals, dim=-1).view(-1).detach()

        pi_taken = torch.gather(pi, dim=1, index=u.reshape(-1,1)).squeeze(1)
        pi_taken[mask == 0] = 1.0
        log_pi_taken =  torch.log(pi_taken)
        coe = self.eval_mixer_net.forward(s).view(-1)

        advantages = (q_taken.view(-1)-baseline).detach()
        coma_loss = -((coe*advantages*log_pi_taken)*mask).sum()/mask.sum()

        self.agent_optimizer.zero_grad()
        coma_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_params, self.args.grad_norm_clip)
        self.agent_optimizer.step()

        # output_targets[avail_u_next == 0.0] = - 9999999
        # q_targets = q_targets.max(dim=3)[0]

        # q_total_eval = self.eval_mixer_net(q_evals, s)
        # q_total_target = self.target_mixer_net(q_targets, s_next)

        # targets = r + self.args.gamma * q_total_target * (1 - terminated)

        # td_error = (q_total_eval - targets.detach())
        # masked_td_error = mask * td_error

        # loss = (masked_td_error ** 2).sum() / mask.sum()
        # self.optimizer.zero_grad()
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        # self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_critic.load_state_dict(self.eval_critic.state_dict())
            self.target_mixer_net.load_state_dict(self.eval_mixer_net.state_dict())

    # def train_critic(self,  )

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
    
    def get_actor_output(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        output_evals, output_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            output_eval = self.eval_actor(inputs, self.eval_hidden)
            output_target = self.target_actor(inputs, self.target_hidden)
            output_eval = output_eval.view(episode_num, self.n_agents, -1)
            output_target = output_target.view(episode_num, self.n_agents, -1)
            output_evals.append(output_eval)
            output_targets.append(output_target)
        output_evals = torch.stack(output_evals, dim=1)
        output_targets = torch.stack(output_targets, dim=1)
        return output_evals, output_targets

    def _get_inputs(self, batch, transition_idx):
        obs, obs_next, u_onehot = batch['o'][:, transition_idx], \
                                  batch['o_next'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)

        if self.args.last_action:
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
            inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1)
        print('input_shape ', inputs.shape)
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                # inputs.
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
            q_eval = self.eval_critic(inputs)  
            q_target = self.target_critic(inputs_next)

            # 把q_eval维度重新变回
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        print('get q values finish')
        return q_evals, q_targets

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        print('Model saved')
        torch.save(self.actor.state_dict(), self.model_dir + '/' + num + '_actor_net_params.pkl')
        torch.save(self.eval_mixer_net.state_dict(), self.model_dir + '/' + num + '_mixer_net_params.pkl')
        torch.save(self.eval_critic.state_dict(),  self.model_dir + '/' + num + '_critic_net_params.pkl')

    def load_model(self, actor_root, critic_root, mixer_root):
        self.actor.load_state_dict(torch.load(actor_root))
        self.eval_critic.load_state_dict(torch.load(critic_root))
        self.eval_mixer_net.load_state_dict(torch.load(mixer_root))
        self.target_critic.load_state_dict(self.eval_critic.state_dict())
        self.target_mixer_net.load_state_dict(self.eval_mixer_net.state_dict()) 