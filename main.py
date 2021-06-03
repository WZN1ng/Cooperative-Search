from common.arguments import get_common_args, get_mixer_args, get_dop_args, get_flight_args, get_reinforce_args,\
                            get_simple_spread_args, get_test_search_args, get_flight_easy_args, get_traditional_args
from env.search_env import SearchEnv
from env.flight_env import FlightSearchEnv
from env.test_env import TestSearchEnv
from env.simple_spread import SimpleSpreadEnv
from env.flight_env_easy import FlightSearchEnvEasy
from runner import Runner
import torch
import pynvml
import os

CIRCLE_DICT_1 = { 'circle_center':[[10,32],[12,15]],
                'circle_radius':[5,7],
                'target_num':[7,8]}

TARGETS_FILENAME = './'

def load_targets(filename):
    f = open(filename, 'r')
    lines = f.readlines()[1:]
    x, y, deter, priority, dx, dy = [], [], [], [], [], []
    for line in lines:
        line = line.split()
        x.append(float(line[0]))
        y.append(float(line[1]))
        deter.append(line[2])
        priority.append(int(line[3]))
        dx.append(float(line[4]))
        dy.append(float(line[5]))
    CIRCLE_DICT = {'x':x, 'y':y, 'deter':deter, 'priority':priority, 'dx':dx, 'dy':dy}
    return CIRCLE_DICT

def smart_gpu_allocate():
    print('Init smart GPU allocation')
    pynvml.nvmlInit()
    gpu_num = pynvml.nvmlDeviceGetCount()
    print('ALL {} gpus list:'.format(gpu_num))
    max_idx = 0
    max_free_mem = 0
    for gpu_idx in range(gpu_num):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        gpu_name = str(pynvml.nvmlDeviceGetName(handle), 'utf-8')
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_mem = meminfo.total
        used_mem = meminfo.used
        free_mem = meminfo.free
        print('{} ({}/{} M)'.format(gpu_name, used_mem//1024**2, total_mem//1024**2))
        if free_mem > max_free_mem:
            max_free_mem = free_mem
            max_idx = gpu_idx
    print('Using GPU {}'.format(max_idx))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(max_idx)
    pynvml.nvmlShutdown()

def get_experiment_seed(args):
    files = os.listdir('./model')
    for f in files:
        tmps = f.split('Seed')
        env = tmps[0][:-1]
        name = args.alg + '_{}a{}t(AM{}TM{})'.format(args.n_agents, args.target_num, args.agent_mode, args.target_mode) 
        if env == args.env and tmps[1].find(name) > -1:
            seed = int(tmps[1].split('_')[0])
            return seed
    return -1

if __name__ == "__main__":
    args = get_common_args()
    if not args.show:
        # thread limit
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        # GPU1
        # if args.cuda:
        #     smart_gpu_allocate()
    if args.show or args.load_model:
        seed = get_experiment_seed(args)
        if seed != -1:
            args.seed = seed

    # load algorithm arguements
    if args.alg.find('qmix') > -1 or args.alg.find('vdn') > -1:
        args = get_mixer_args(args)
    elif args.alg.find('dop') > -1:
        args = get_dop_args(args)
    elif args.alg.find('reinforce') > -1:
        args = get_reinforce_args(args)
    elif args.alg.find('random') > -1:
        args = get_traditional_args(args)

    # load environment arguements
    if args.env == 'search':
        args = get_test_search_args(args)
        env = SearchEnv(args, circle_dict=CIRCLE_DICT_1)
    elif args.env == 'flight':
        args = get_flight_args(args)
        targets_root = './flight_targets.txt'
        CIRCLE_DICT_2 = load_targets(targets_root)
        env = FlightSearchEnv(args, circle_dict=CIRCLE_DICT_2)
    elif args.env == 'flight_easy':
        args = get_flight_easy_args(args)
        targets_root = './flight_targets.txt'
        CIRCLE_DICT_2 = load_targets(targets_root)
        env = FlightSearchEnvEasy(args, circle_dict=CIRCLE_DICT_2)
    elif args.env == 'test':
        args = get_flight_args(args)
        env = TestSearchEnv()
    elif args.env == 'simple_spread':
        args = get_simple_spread_args(args)
        env = SimpleSpreadEnv(args)


    env_info = env.get_env_info()
    args.n_actions = env_info['n_actions']
    args.state_shape = env_info['state_shape']
    args.obs_shape = env_info['obs_shape']
    args.episode_limit = env_info['episode_limit']
    # print(args.n_actions, args.obs_shape, args.state_shape)

    # run experiments
    for i in range(1):
        runner = Runner(env, args)
        if args.show:
            if not args.experiment:
                runner.replay(20)
            else:
                model_idx = 25
                replay_times = 100
                print('using model {}, replay time {}'.format(model_idx, replay_times))
                runner.collect_experiment_data(num=model_idx, replay_times=replay_times)
        else:
            runner.run(i)
        env.close()
