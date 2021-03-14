from common.arguments import get_common_args, get_mixer_args, get_dop_args, get_flight_args
from env.search_env import SearchEnv
from env.flight_env import FlightSearchEnv
from runner import Runner

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

if __name__ == "__main__":
    for i in range(8):
        args = get_common_args()
        if args.alg.find('qmix') > -1:
            args = get_mixer_args(args)
        elif args.alg.find('dop') > -1:
            args = get_dop_args(args)
        if args.env == 'search':
            env = SearchEnv(args, circle_dict=CIRCLE_DICT_1)
        elif args.env == 'flight':
            args = get_flight_args(args)
            targets_root = './flight_targets.txt'
            CIRCLE_DICT_2 = load_targets(targets_root)
            env = FlightSearchEnv(args, circle_dict=CIRCLE_DICT_2)
        env_info = env.get_env_info()
        args.n_actions = env_info['n_actions']
        args.state_shape = env_info['state_shape']
        args.obs_shape = env_info['obs_shape']
        args.episode_limit = env_info['episode_limit']
        runner = Runner(env, args)
        if args.show:
            runner.replay(5)
            break
        elif args.learn:
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
