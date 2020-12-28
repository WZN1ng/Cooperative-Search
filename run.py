from agent import Agent
from target import Target
from env import EnvSearch
from memory import MemoryReplay
from dqn import DQN

import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib import animation
import os

MAP_SIZE = 50
AGENT_NUM = 1
VIEW_RANGE = 6
TARGET_NUM = 20
ENV_MODE = 2

BATCH_SIZE = 32
NUM_EPISODES = 500
MAx_STEP = 300
GAMMA = 0.9
REWARD_THRES = -1000

# initial agent
def initial_agents(map_size, agent_num, view_range, mode):
    agent_list = []
    if mode == 1:               # mode == 1 在地图中央生成agents
        length_square = int(np.ceil(np.sqrt(agent_num)))
        start_pos = [(map_size-length_square)//2 for _ in range(2)]
        for i in range(start_pos[0], start_pos[0]+length_square):
            if len(agent_list) == agent_num:
                break
            for j in range(start_pos[0], start_pos[0]+length_square):
                if len(agent_list) == agent_num:
                    break
                temp_agent = Agent([i,j], view_range, map_size)
                agent_list.append(temp_agent)
    elif mode == 2:             # mode == 2 左下角生成agents
        length_square = int(np.ceil(np.sqrt(agent_num)))
        start_pos = [map_size-1, 0]
        for i in range(start_pos[0], start_pos[0]-length_square, -1):
            if len(agent_list) == agent_num:
                break
            for j in range(start_pos[1], start_pos[1]+length_square):
                if len(agent_list) == agent_num:
                    break
                temp_agent = Agent([i,j], view_range, map_size)
                agent_list.append(temp_agent)
    # print(len(agent_list))
    return agent_list

# 运行
def run(agent_list, env, dict):
    # for i_episode in range(NUM_EPISODES):
    env.reset(ENV_MODE, dict)
    pos_list = []
    obs_list = []

    for i,agent in enumerate(agent_list):
        agent.reset()
        env.get_agent_obs(agent, i) # 初始化观测阵
        pos_list.append(agent.pos)

    while True:
        obs_list = env.obs_list.copy()
        act_list = [a.select_action(1, obs_list[i]) for i,a in enumerate(agent_list)]
        done = env.step(agent_list, act_list)
        env.render(agent_list)
        if done or env.total_reward <= REWARD_THRES:
            break

# train
def train(agent_list, env, dict):
    score_list = []
    best_model = 0
    for i_episode in range(NUM_EPISODES):
        # 重置环境与智能体
        env.reset(ENV_MODE, dict)
        pos_list = []
        obs_list = []
        for i,agent in enumerate(agent_list):
            agent.reset()
            env.get_agent_obs(agent, i) # 初始化观测阵
            pos_list.append(agent.pos)

        # 开始训练
        for j in range(MAx_STEP):
            obs_list = env.obs_list.copy()
            act_list = [a.select_action(0, obs_list[i]) for i,a in enumerate(agent_list)]
            # print(act_list)
            # for agent in agent_list:        # 存记忆
            #     s = agent.cumu_obs
            #     if s.size != 0:
            #         # print(s.shape)
            #         s_list.append(s)

            done = env.step(agent_list, act_list)

            for i,agent in enumerate(agent_list):   # 存经验
                next_obs = env.obs_list[i]
                next_pos = agent.pos
                r = env.curr_reward
                agent.store_transition(obs_list[i], pos_list[i], act_list[i], next_obs, next_pos, r)
            
            # 经验池满后开始训练
            if len(agent.memory) >= BATCH_SIZE:
                for agent in agent_list:
                    agent.model.learn()

            if done:    # 结束判断
                break
        
            # env.render(agent_list)
        score_list.append(env.total_reward)

        if env.target_find > best_model:
            best_model = env.target_find
            save_model_params(agent_list, env.target_find)

        if i_episode % 10 == 0:
            print('{} total_reward:{:.1f}  targets:{}/{}'.format(i_episode, env.total_reward, 
                    env.target_find, env.target_num))

    # save_model_params(agent_list)   # 保存模型

    # 输出结果
    plt.cla()
    plt.plot(score_list)
    plt.savefig('train.jpg')

# 保存模型参数
def save_model_params(agent_list, target_find):
    root = os.getcwd()+'/models/'
    folder = '{}X{}_{}targets_{}agents'.format(MAP_SIZE,MAP_SIZE,TARGET_NUM,AGENT_NUM)
    root = root + folder + '/'
    if not os.path.exists(root):
        os.makedirs(root)
    for i,agent in enumerate(agent_list):
        fileroot1 = root+'eval_model_'+str(i)+'_'+str(target_find)+'tf'+'.pkl'
        fileroot2 = root+'target_model_'+str(i)+'_'+str(target_find)+'tf'+'.pkl'
        agent.save_model(fileroot1, fileroot2)

# 
def show(agent_list, env, dic, tf):
    root = './models/50X50_20targets_1agents/'
    files = os.listdir(root)
    for index,agent in enumerate(agent_list):
        eva, tar = '', ''
        for f in files:
            parts = f.split('.')[0].split('_')  
            if parts[2] == str(index) and parts[3] == str(tf)+'tf':
                if parts[0] == 'eval':
                    eva = f
                elif parts[0] == 'target':
                    tar = f
                if eva != '' and tar != '':
                    print(eva, tar)
                    break
        agent.load_model(root+eva, root+tar)
    run(agent_list, env, dic)


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    agent_list = initial_agents(MAP_SIZE, AGENT_NUM, VIEW_RANGE, 2)     # 生成智能体
    dic = {'circle_num':20, 'circle_center':[[4,44],[40,41],[6,3]],
            'circle_radius':[12,12,8],'target_num':[6,7,7],'filename':'targets2.txt'}
    env = EnvSearch(MAP_SIZE, TARGET_NUM, ENV_MODE, dic)      # 生成环境
    # env.save_target('targets2.txt')   # 保存目标

    # 训练模型
    # train(agent_list, env, dict)

    # 展示模型
    show(agent_list, env, dic, 19)



