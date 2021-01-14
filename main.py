from common.arguments import get_common_args, get_mixer_args
from env.search_env import SearchEnv
from runner import Runner

CIRCLE_DICT = { 'circle_center':[[10,32],[12,15]],
                'circle_radius':[5,7],
                'target_num':[7,8]}
TARGETS_FILENAME = './'

if __name__ == "__main__":
    for i in range(8):
        args = get_common_args()
        if args.alg.find('qmix') > -1:
            args = get_mixer_args(args)
        env = SearchEnv(args, circle_dict=CIRCLE_DICT)
        env_info = env.get_env_info()
        args.n_actions = env_info['n_actions']
        args.state_shape = env_info['state_shape']
        args.obs_shape = env_info['obs_shape']
        args.episode_limit = env_info['episode_limit']
        runner = Runner(env, args)
        if args.show:
            runner.replay(16)
            break
        elif args.learn:
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
