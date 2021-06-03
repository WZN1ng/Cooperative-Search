import argparse
import numpy as np
import os

"""
Here are the params for the training

"""
SEED = [19990227, 19991023, 19980123, 19990417]

def get_current_seeds():
    root = './result'
    files = os.listdir(root)
    seeds = []
    for f in files:
        tmps = f.split('_')
        for tmp in tmps:
            if tmp.find('Seed') > -1:
                seeds.append(int(tmp.split('Seed')[1]))
    # print(seeds)
    return seeds

def get_common_args():
    parser = argparse.ArgumentParser()
    
    # environment settings 
    parser.add_argument('--env', type=str, default='flight_easy', help='the environment of the experiment')
    parser.add_argument('--map_size', type=int, default=50, help='the size of the grid map')
    parser.add_argument('--target_num', type=int, default=15, help='the num of the search targets')
    parser.add_argument('--target_mode', type=int, default=0, help='targets location mode')
    parser.add_argument('--target_dir', type=str, default='./targets/', help='targets directory')
    parser.add_argument('--agent_mode', type=int, default=0, help='agents location mode')       # bottom line
    parser.add_argument('--n_agents', type=int, default=3, help='the num of agents')
    parser.add_argument('--view_range', type=int, default=7, help='the view range of agent')

    # algorithms args
    parser.add_argument('--alg', type=str, default='reinforce', help='the algorithms for training')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use common network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
    parser.add_argument('--evaluate_epoch', type=int, default=20, help='number of the epoch to evaluate the agents')
    parser.add_argument('--model_dir', type=str, default='./model/', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result/', help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    parser.add_argument('--seed_idx', type=int, default=4, help='the index of the model seed list')

    # train or show model
    parser.add_argument('--show', type=bool, default=True, help='train or show model')
    parser.add_argument('--experiment', type=bool, default=False, help='whether to collect experimental data for certain algorithm')
    args = parser.parse_args()
    return args

# arguments of qmix
def get_mixer_args(args):
    # network 
    args.off_policy = True
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.lr = 0.0005
    if not args.show and not args.load_model:
        if args.seed_idx < len(SEED):
            args.seed = SEED[args.seed_idx]
        else:
            currseeds = get_current_seeds()
            while True:
                seed = np.random.randint(10000000, 99999999)
                if seed not in currseeds:
                    args.seed = seed
                    break
    args.tau = 0.05

    # epsilon greedy
    # args.epsilon_random = 500
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 10000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon)/anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the epoch for training
    args.n_epoch = 500000

    # the number of the episodes in one epoch
    args.n_episodes = 1

    # the number of the train step in one epoch
    args.train_steps = 1

    # how often to evaluate
    args.evaluate_cycle = 200

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(3000)

    # how often to save the model
    args.save_cycle = 500

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args

# args of dop
def get_dop_args(args):
    # network
    args.off_policy = True
    args.rnn_hidden_dim = 64
    args.offpg_hidden_dim = 128
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.lr = 5e-4
    args.critic_lr = 1e-4
    args.td_lambda = 0.8
    
    if not args.show and not args.load_model:
        if args.seed_idx < len(SEED):
            args.seed = SEED[args.seed_idx]
        else:
            currseeds = get_current_seeds()
            while True:
                seed = np.random.randint(10000000, 99999999)
                if seed not in currseeds:
                    args.seed = seed
                    break
    args.tau = 0.05

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 10000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon)/anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the epoch for training
    args.n_epoch = 500000

    # the number of the episodes in one epoch
    args.n_episodes = 1

    # the number of the train step in one epoch
    args.train_steps = 1

    # how often to evaluate
    args.evaluate_cycle = 200

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(3000)
    args.onbuffer_size = int(32)

    # how often to save the model
    args.save_cycle = 500

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args

# arguments of central_v
def get_reinforce_args(args):
    # network
    args.off_policy = False
    args.rnn_hidden_dim = 64
    args.critic_dim = 128
    if not args.show and not args.load_model:
        if args.seed_idx < len(SEED):
            args.seed = SEED[args.seed_idx]
        else:
            currseeds = get_current_seeds()
            while True:
                seed = np.random.randint(10000000, 99999999)
                if seed not in currseeds:
                    args.seed = seed
                    break
    args.tau = 0.05
    args.lr_actor = 1e-4
    args.lr_critic = 1e-3

    # epsilon-greedy
    args.epsilon = 0.5
    args.anneal_epsilon = 0.00064
    args.min_epsilon = 0.02
    args.epsilon_anneal_scale = 'epoch'

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(1000)

    # the number of the epoch to train the agent
    args.n_epoch = 100000

    # the number of the episodes in one epoch
    args.n_episodes = 1

    # how often to evaluate
    args.evaluate_cycle = 200

    # how often to save the model
    args.save_cycle = 500

    # prevent gradient explosion
    args.grad_norm_clip = 10

    return args

# args of test search env
def get_test_search_args(args):
    args.conv = False
    args.search_env = True

    return args

# args of simple spread
def get_simple_spread_args(args):
    args.conv = False
    args.search_env = False

    return args

# args of flight search env
def get_flight_args(args):
    # flight info
    args.agent_velocity = 1
    args.time_limit = 200      
    args.turn_limit = np.pi/4           # rad/s
    args.flight_height = 8000     
    args.safe_dist = 1
    args.detect_prob = 0.9
    args.wrong_alarm_prob = 0.1
    args.force_dist = 3
    args.search_env = True

    # env
    args.conv = True
    # args.dim_1 = 4
    # args.kernel_size_1 = 4
    # args.stride_1 = 2
    # # args.padding = 2
    # args.dim_2 = 1
    # args.kernel_size_2 = 3
    # args.stride_2 = 1
    # args.padding_2 = 1

    args.dim_1 = 4
    args.kernel_size_1 = 4
    args.stride_1 = 2
    # args.padding = 2
    args.dim_2 = 1
    args.kernel_size_2 = 3
    args.stride_2 = 1
    args.padding_2 = 1

    args.conv_out_dim = 16

    return args

def get_flight_easy_args(args):
    # flight info
    args.agent_velocity = 1
    args.time_limit = 200      
    args.turn_limit = np.pi/4           # rad/s
    args.flight_height = 8000     
    args.safe_dist = 1
    args.detect_prob = 0.9
    args.wrong_alarm_prob = 0.1
    args.force_dist = 3
    args.search_env = True

    # env
    args.conv = False

    return args

def get_traditional_args(args):
    args.off_policy = False
    args.epsilon = 0
    args.anneal_epsilon = 0
    args.min_epsilon = 0

    args.seed = 0
    return args

