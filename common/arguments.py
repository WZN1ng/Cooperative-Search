import argparse

"""
Here are the params for the training

"""

def get_common_args():
    parser = argparse.ArgumentParser()
    
    # environment settings 
    parser.add_argument('--map_size', type=int, default=50, help='the size of the grid map')
    parser.add_argument('--target_num', type=int, default=15, help='the num of the search targets')
    parser.add_argument('--target_mode', type=int, default=2, help='targets location mode')
    parser.add_argument('--target_dir', type=str, default='./targets/', help='targets directory')
    parser.add_argument('--agent_mode', type=int, default=2, help='agents location mode')
    parser.add_argument('--n_agents', type=int, default=3, help='the num of agents')
    parser.add_argument('--view_range', type=int, default=6, help='the view range of agent')

    # algorithms args
    parser.add_argument('--alg', type=str, default='qmix', help='the algorithms for training')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use common network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--optimizer', type=str, default='RMS', help='optimizer')
    parser.add_argument('--evaluate_epoch', type=int, default=20, help='number of the epoch to evaluate the agents')
    parser.add_argument('--model_dir', type=str, default='./model/', help='model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./result/', help='result directory of the policy')
    parser.add_argument('--load_model', type=bool, default=True, help='whether to load the pretrained model')
    parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
    parser.add_argument('--cuda', type=bool, default=True, help='whether to use the GPU')
    
    # train or show model
    parser.add_argument('--show', type=bool, default=True, help='train or show model')
    args = parser.parse_args()
    return args

# arguments of qmix
def get_mixer_args(args):
    # network 
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.two_hyper_layers = False
    args.hyper_hidden_dim = 64
    args.lr = 5e-4

    # epsilon greedy
    args.epsilon = 1
    args.min_epsilon = 0.05
    anneal_steps = 50000
    args.anneal_epsilon = (args.epsilon - args.min_epsilon)/anneal_steps
    args.epsilon_anneal_scale = 'step'

    # the number of the epoch for training
    args.n_epoch = 20000

    # the number of the episodes in one epoch
    args.n_episodes = 1

    # the number of the train step in one epoch
    args.train_steps = 1

    # how often to evaluate
    args.evaluate_cycle = 100

    # experience replay
    args.batch_size = 32
    args.buffer_size = int(100)

    # how often to save the model
    args.save_cycle = 500

    # how often to update the target_net
    args.target_update_cycle = 200

    # prevent gradient explosion
    args.grad_norm_clip = 10

    # load model
    if args.load_model:
        args.model_index = 16

    return args

