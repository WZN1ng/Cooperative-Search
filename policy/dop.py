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
        self.seed = args.seed
        self.tau = args.tau
        self.critic_training_steps = 0
        self.last_target_update_step = 0
        input_shape = self.obs_shape

        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents
        critic_input_shape = self.state_shape + self.obs_shape + self.n_agents

        # random seed 
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        
        # DEBUG
        # torch.autograd.set_detect_anomaly(True)

        # nets
        self.actor = RNN(input_shape, args)
        self.eval_critic = OffPGCritic(critic_input_shape, args)
        self.target_critic = OffPGCritic(critic_input_shape, args)
        self.eval_mixer_net = MixerNet(args)
        self.target_mixer_net = MixerNet(args)
        if self.args.cuda:
            self.actor.cuda()
            self.eval_critic.cuda()
            self.target_critic.cuda()
            self.eval_mixer_net.cuda()
            self.target_mixer_net.cuda()
        self.model_dir = args.model_dir + args.env + '_Seed' + str(args.seed) + '_' + args.alg + \
                        '_{}a{}t(AM{}TM{})'.format(args.n_agents, args.target_num, args.agent_mode, args.target_mode)

        # load model
        if self.args.load_model:
            model_index = self.get_model_idx() - 1
            if os.path.exists(os.path.join(self.model_dir, str(model_index) + '_actor_net_params.pkl')):
                path_rnn = os.path.join(self.model_dir, str(model_index) + '_actor_net_params.pkl')
                path_critic = os.path.join(self.model_dir, str(model_index) + '_critic_net_params.pkl')
                path_mixer = os.path.join(self.model_dir, str(model_index) + '_mixer_net_params.pkl')
                map_location = 'cuda' if self.args.cuda else 'cpu'
                self.actor.load_state_dict(torch.load(path_rnn, map_location=map_location))
                self.eval_critic.load_state_dict(torch.load(path_critic, map_location=map_location))
                self.eval_mixer_net.load_state_dict(torch.load(path_mixer, map_location=map_location))
                print('Successfully load the model: {}, {} and {}'.format(path_rnn, path_critic, path_mixer))
            else:
                raise Exception('No such model')

        # update target net params
        self.target_critic.load_state_dict(self.eval_critic.state_dict())
        self.target_mixer_net.load_state_dict(self.eval_mixer_net.state_dict())

        self.actor_params = list(self.actor.parameters())
        self.critic_params = list(self.eval_critic.parameters())
        self.mixer_params = list(self.eval_mixer_net.parameters())
        self.params = self.actor_params + self.critic_params
        self.c_params = self.critic_params + self.mixer_params 

        if args.optimizer == 'RMS':
            self.agent_optimizer = torch.optim.RMSprop(self.actor_params, lr=args.lr)
            self.critic_optimizer = torch.optim.RMSprop(self.critic_params, lr=args.critic_lr)
            self.mixer_optimizer = torch.optim.RMSprop(self.mixer_params, lr=args.critic_lr)
        elif args.optimizer == 'Adam':
            self.agent_optimizer = torch.optim.Adam(self.actor_params, lr=args.lr)
            self.critic_optimizer = torch.optim.Adam(self.critic_params, lr=args.critic_lr)
            self.mixer_optimizer = torch.optim.Adam(self.mixer_params, lr=args.critic_lr)
        else:
            raise Exception('No such optimizer')

        self.eval_hidden = None
        print('Init alg DOP(seed = {})'.format(self.seed))

    def learn(self, batch, max_episode_len, train_step, epsilon):
        # print('learn')
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
        mask = (1 - batch['padded'].float()).repeat(1, 1, self.n_agents)
        # print('padded: ', batch['padded'].shape)
        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
        
        q_values = self._train_critic(batch, max_episode_len, train_step)
        action_prob = self._get_actor_output(batch, max_episode_len, epsilon)

        # baseline
        # print('q_vals :', q_vals.shape, '  u :', u.shape)
        q_taken = torch.gather(q_values, dim=3, index=u).squeeze(3)
        pi_taken = torch.gather(action_prob, dim=3, index=u).squeeze(3)
        pi_taken[mask == 0] = 1.0
        log_pi_taken = torch.log(pi_taken)

        # pi = actor_out.view(-1, self.n_actions)
        baseline = (q_values * action_prob).sum(dim=3, keepdim=True).squeeze(3).detach()
        advantage = (q_taken - baseline).detach()

        coma_loss = - ((advantage * log_pi_taken) * mask).sum() / mask.sum()

        self.agent_optimizer.zero_grad()
        coma_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_params, self.args.grad_norm_clip)
        self.agent_optimizer.step()

    def soft_update(self):
        for param, target_param in zip(self.eval_critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)
        for param, target_param in zip(self.eval_mixer_net.parameters(), self.target_mixer_net.parameters()):
            target_param.data.copy_(self.tau*param.data + (1-self.tau)*target_param.data)

    def _train_critic(self, batch, max_episode_len, train_step):
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                             batch['r'],  batch['avail_u'], batch['avail_u_next'],\
                                                             batch['terminated']
        u_next = u[:, 1:]
        padded_u_next = torch.zeros(*u[:, -1].shape, dtype=torch.long).unsqueeze(1)
        u_next = torch.cat((u_next, padded_u_next), dim=1)
        mask = (1 - batch["padded"].float()).repeat(1, 1, self.n_agents)
        
        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            u_next = u_next.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()

        q_evals, q_targets = self._get_q_values(batch, max_episode_len)
        q_values = q_evals.clone()

        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        q_targets = torch.gather(q_targets, dim=3, index=u_next).squeeze(3)
        
        q_total_eval = self.eval_mixer_net(q_evals, s)
        q_total_target = self.target_mixer_net(q_targets, s_next)

        targets = self._td_lambda_target(batch, max_episode_len, q_total_target.cpu())
        if self.args.cuda:
            targets = targets.cuda()
        
        td_error = targets.detach() - q_total_eval
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        critic_loss = (masked_td_error ** 2).sum() / mask.sum()

        self.critic_optimizer.zero_grad()
        self.mixer_optimizer.zero_grad()
        critic_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.c_params, self.args.grad_norm_clip)
        self.critic_optimizer.step()
        self.mixer_optimizer.step()

        # if train_step > 0 and train_step % self.args.target_update_cycle == 0:
        #     self.target_critic.load_state_dict(self.eval_critic.state_dict())
        #     self.target_mixer_net.load_state_dict(self.eval_mixer_net.state_dict())
        self.soft_update()
        return q_values

    def _td_lambda_target(self, batch, max_episode_len, q_targets):
        # batch.shep = (episode_num, max_episode_len， n_agents，n_actions)
        # q_targets.shape = (episode_num, max_episode_len， n_agents)
        episode_num = batch['o'].shape[0]
        mask = (1 - batch["padded"].float()).repeat(1, 1, self.args.n_agents)
        terminated = (1 - batch["terminated"].float()).repeat(1, 1, self.args.n_agents)
        r = batch['r'].repeat((1, 1, self.args.n_agents))
        # --------------------------------------------------n_step_return---------------------------------------------------
        '''
        1. 每条经验都有若干个n_step_return，所以给一个最大的max_episode_len维度用来装n_step_return
        最后一维,第n个数代表 n+1 step。
        2. 因为batch中各个episode的长度不一样，所以需要用mask将多出的n-step return置为0，
        否则的话会影响后面的lambda return。第t条经验的lambda return是和它后面的所有n-step return有关的，
        如果没有置0，在计算td-error后再置0是来不及的
        3. terminated用来将超出当前episode长度的q_targets和r置为0
        '''
        n_step_return = torch.zeros((episode_num, max_episode_len, self.args.n_agents, max_episode_len))
        for transition_idx in range(max_episode_len - 1, -1, -1):
            # 最后计算1 step return
            n_step_return[:, transition_idx, :, 0] = (r[:, transition_idx] + self.args.gamma * q_targets[:, transition_idx] * terminated[:, transition_idx]) * mask[:, transition_idx]        # 经验transition_idx上的obs有max_episode_len - transition_idx个return, 分别计算每种step return
            # 同时要注意n step return对应的index为n-1
            for n in range(1, max_episode_len - transition_idx):
                # t时刻的n step return =r + gamma * (t + 1 时刻的 n-1 step return)
                # n=1除外, 1 step return =r + gamma * (t + 1 时刻的 Q)
                n_step_return[:, transition_idx, :, n] = (r[:, transition_idx] + self.args.gamma * n_step_return[:, transition_idx + 1, :, n - 1]) * mask[:, transition_idx]
        # --------------------------------------------------n_step_return---------------------------------------------------

        # --------------------------------------------------lambda return---------------------------------------------------
        '''
        lambda_return.shape = (episode_num, max_episode_len，n_agents)
        '''
        lambda_return = torch.zeros((episode_num, max_episode_len, self.args.n_agents))
        for transition_idx in range(max_episode_len):
            returns = torch.zeros((episode_num, self.args.n_agents))
            for n in range(1, max_episode_len - transition_idx):
                returns += pow(self.args.td_lambda, n - 1) * n_step_return[:, transition_idx, :, n - 1]
            lambda_return[:, transition_idx] = (1 - self.args.td_lambda) * returns + \
                                            pow(self.args.td_lambda, max_episode_len - transition_idx - 1) * \
                                            n_step_return[:, transition_idx, :, max_episode_len - transition_idx - 1]
        # --------------------------------------------------lambda return---------------------------------------------------
        return lambda_return

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
    
    def _get_actor_output(self, batch, max_episode_len, epsilon):
        episode_num = batch['o'].shape[0]
        avail_actions = batch['avail_u']  # coma不用target_actor，所以不需要最后一个obs的下一个可执行动作
        action_prob = []
        for transition_idx in range(max_episode_len):
            inputs = self._get_actor_inputs(batch, transition_idx)  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
            outputs, self.eval_hidden = self.actor(inputs, self.eval_hidden)  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            # 把q_eval维度重新变回(8, 5,n_actions)
            outputs = outputs.view(episode_num, self.n_agents, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)
            action_prob.append(prob)
        # 得的action_prob是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        action_prob = torch.stack(action_prob, dim=1).cpu()

        action_num = avail_actions.sum(dim=-1, keepdim=True).float().repeat(1, 1, 1, avail_actions.shape[-1])   # 可以选择的动作的个数
        action_prob = ((1 - epsilon) * action_prob + torch.ones_like(action_prob) * epsilon / action_num)
        action_prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        # 因为上面把不能执行的动作概率置为0，所以概率和不为1了，这里要重新正则化一下。执行过程中Categorical会自己正则化。
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)
        # 因为有许多经验是填充的，它们的avail_actions都填充的是0，所以该经验上所有动作的概率都为0，在正则化的时候会得到nan。
        # 因此需要再一次将该经验对应的概率置为0
        action_prob[avail_actions == 0] = 0.0
        if self.args.cuda:
            action_prob = action_prob.cuda()
        return action_prob
    
    def _get_actor_inputs(self, batch, transition_idx):
        obs, u_onehot = batch['o'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs = []
        inputs.append(obs)

        if self.args.last_action:
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx-1])
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        inputs = torch.cat([x.reshape(episode_num*self.args.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_critic_inputs(self, batch, transition_idx, max_episode_len):
        obs, obs_next, s, s_next = batch['o'][:, transition_idx], batch['o_next'][:, transition_idx],\
                                   batch['s'][:, transition_idx], batch['s_next'][:, transition_idx]
        u_onehot = batch['u_onehot'][:, transition_idx]
        if transition_idx != max_episode_len - 1:
            u_onehot_next = batch['u_onehot'][:, transition_idx + 1]
        else:
            u_onehot_next = torch.zeros(*u_onehot.shape)
        
        s = s.unsqueeze(1).expand(-1, self.n_agents, -1)
        s_next = s_next.unsqueeze(1).expand(-1, self.n_agents, -1)
        episode_num = obs.shape[0]

        u_onehot = u_onehot.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)
        u_onehot_next = u_onehot_next.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)

        if transition_idx == 0:
            u_onehot_last = torch.zeros_like(u_onehot)
        else:
            u_onehot_last = batch['u_onehot'][:, transition_idx - 1]
            u_onehot_last = u_onehot_last.view((episode_num, 1, -1)).repeat(1, self.n_agents, 1)

        inputs, inputs_next = [], []
        inputs.append(s)
        inputs_next.append(s_next)
        inputs.append(obs)
        inputs_next.append(obs_next)
        # inputs.append(u_onehot)
        # inputs_next.append(u_onehot_next)
        # # if self.args.last_action:
        # inputs.append(u_onehot_last)
        # inputs_next.append(u_onehot)
        # # if self.args.reuse_network:
        inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        inputs_next.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        inputs = torch.cat([x.reshape(episode_num*self.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num*self.n_agents, -1) for x in inputs_next], dim=1)
        return inputs, inputs_next

    def _get_q_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_critic_inputs(batch, transition_idx, max_episode_len)  # 给obs加last_action、agent_id
            if self.args.cuda:
                # inputs.
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
            q_eval = self.eval_critic(inputs)  
            q_target = self.target_critic(inputs_next)
            # print('get q value', inputs.shape, q_eval.shape)

            # 把q_eval维度重新变回
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        # print('get q values finish')
        return q_evals, q_targets

    def save_model(self, num):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        print('Model saved')
        idx = str(num)

        torch.save(self.actor.state_dict(), os.path.join(self.model_dir, idx + '_actor_net_params.pkl'))
        torch.save(self.eval_mixer_net.state_dict(), os.path.join(self.model_dir, idx + '_mixer_net_params.pkl'))
        torch.save(self.eval_critic.state_dict(), os.path.join(self.model_dir, idx + '_critic_net_params.pkl'))

    def load_model(self, actor_root, critic_root, mixer_root):
        self.actor.load_state_dict(torch.load(actor_root))
        self.eval_critic.load_state_dict(torch.load(critic_root))
        self.eval_mixer_net.load_state_dict(torch.load(mixer_root))
        self.target_critic.load_state_dict(self.eval_critic.state_dict())
        self.target_mixer_net.load_state_dict(self.eval_mixer_net.state_dict()) 

    def get_model_idx(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            return 0
        idx = 0
        models = os.listdir(self.model_dir)
        for model in models:
            num = int(model.split('_')[0])
            idx = max(idx, num)
        idx += 1
        return idx