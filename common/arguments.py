import argparse
import numpy as np

"""
Here are the params for the training

"""
SEED = [27, 10, 12]

def get_common_args():
    parser = argparse.ArgumentParser()
    
    # environment settings 
    parser.add_argument('--env', type=str, default='flight', help='the environment of the experiment')
    parser.add_argument('--map_size', type=int, default=100, help='the size of the grid map')
    parser.add_argument('--target_num', type=int, default=15, help='the num of the search targets')
    parser.add_argument('--target_mode', type=int, default=2, help='targets location mode')
    parser.add_argument('--target_dir', type=str, default='./targets/', help='targets directory')
    parser.add_argument('--agent_mode', type=int, default=1, help='agents location mode')       # bottom line
    parser.add_argument('--n_agents', type=int, default=3, help='the num of agents')
    parser.add_argument('--view_range', type=int, default=10, help='the view range of agent')

    # algorithms args
    parser.add_argument('--alg', type=str, default='qmix', help='the algorithms for training')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use common network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default='RMS', help='optimizer')
    parser.add_argument('--evaluate_epoch', type=int, default=20, help='number of the epoch to evaluate the agents')
    parser.add_argument('--model_dir', type=str, default='./model/', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result/', help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    parser.add_argument('--seed_idx', type=int, default=0, help='the index of the model seed list')

    # train or show model
    parser.add_argument('--show', type=bool, default=False, help='train or show model')
    args = parser.parse_args()
    return args

# arguments of qmix
def get_mixer_args(args):
    # network 
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.lr = 0.01
    args.seed = SEED[args.seed_idx]
    args.tau = 0.005

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 100000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon)/anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the epoch for training
    args.n_epoch = 10000

    # the number of the episodes in one epoch
    args.n_episodes = 1

    # the number of the train step in one epoch
    args.train_steps = 1

    # how often to evaluate
    args.evaluate_cycle = 100

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(1000)

    # how often to save the model
    args.save_cycle = 500

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # load model
    if args.load_model:
        args.model_index = 0

    return args

# args of dop
def get_dop_args(args):
    # network
    args.rnn_hidden_dim = 64
    args.offpg_hidden_dim = 256
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.lr = 5e-4
    args.critic_lr = 1e-4

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon)/anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the epoch for training
    args.n_epoch = 10000

    # the number of the episodes in one epoch
    args.n_episodes = 1

    # the number of the train step in one epoch
    args.train_steps = 1

    # how often to evaluate
    args.evaluate_cycle = 100

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(1000)

    # how often to save the model
    args.save_cycle = 500

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # load model
    if args.load_model:
        args.model_index = 0

    return args

# args of flight search env
def get_flight_args(args):
    # flight info
    args.agent_velocity = 1
    args.time_limit = 1000      
    args.turn_limit = np.pi/4           # rad/s
    args.flight_height = 8000     
    args.safe_dist = 1
    args.detect_prob = 0.9
    args.force_dist = 3

    return args

