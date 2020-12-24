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
NUM_EPISODES = 2000
GAMMA = 0.9
REWARD_THRES = -100

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
    return agent_list

# 运行
def run(agent_list, env, dict):
    # for i_episode in range(NUM_EPISODES):
    env.reset(3, dict)
    while True:
        act_list = [a.select_action(1) for a in agent_list]
        done = env.step(agent_list, act_list)
        env.render(agent_list)
        if done or env.total_reward <= REWARD_THRES:
            break

# train
def train(agent_list, env, dict):
    score_list = []
    for i_episode in range(NUM_EPISODES):
        # 重置环境与智能体
        env.reset(ENV_MODE, dict)
        for agent in agent_list:
            agent.reset()

        # 开始训练
        while True:
            act_list = [a.select_action(0) for a in agent_list]
            # print(act_list)
            s_list = []
            for agent in agent_list:        # 存记忆
                s = agent.cumu_obs
                if s.size != 0:
                    # print(s.shape)
                    s_list.append(s)

            done = env.step(agent_list, act_list)
            if len(s_list) == AGENT_NUM:                # 存记忆
                for i,agent in enumerate(agent_list):
                    next_s = agent.cumu_obs
                    r = env.curr_reward
                    agent.memory.push(s_list[i], act_list[i], next_s, r)
            
            # 经验池满后开始训练
            if agent.memory.is_full():
                for agent in agent_list:
                    agent.model.learn()

            if done or env.total_reward <= REWARD_THRES:    # 结束判断
                break
        
            env.render(agent_list)
        score_list.append(env.total_reward)

        if i_episode % 100 == 0:
            print('{} total_reward:{:.1f}  targets:{}/{}'.format(i_episode, env.total_reward, 
                    env.target_find, env.target_num))

    save_model_params(agent_list)   # 保存模型

    # 输出结果
    plt.cla()
    plt.plot(score_list)

# 保存模型参数
def save_model_params(agent_list):
    root = os.getcwd()+'/models/'
    folder = '{}X{}_{}targets_{}agents'.format(MAP_SIZE,MAP_SIZE,TARGET_NUM,AGENT_NUM)
    root = root + folder + '/'
    if not os.path.exists(root):
        os.makedirs(root)
    for i,agent in enumerate(agent_list):
        fileroot1 = root+'eval_model_'+str(i)+'.pkl'
        fileroot2 = root+'target_model_'+str(i)+'.pkl'
        agent.save_model(fileroot1, fileroot2)

if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    agent_list = initial_agents(MAP_SIZE, AGENT_NUM, VIEW_RANGE, 1)     # 生成智能体
    dict = {'circle_num':20, 'circle_center':[[4,44],[40,41],[42,3]],
            'circle_radius':[12,12,8],'target_num':[6,7,7],'filename':'targets2.txt'}
    env = EnvSearch(MAP_SIZE, TARGET_NUM, ENV_MODE, dict)      # 生成环境
    # env.save_target('targets2.txt')   # 保存目标

    # 训练模型
    # train(agent_list, env, dict)

    # 展示模型
    root = './models/50X50_20targets_1agents/'
    for i,agent in enumerate(agent_list):
        filename1 = 'eval_model_'+str(i)+'.pkl'
        filename2 = 'target_model_'+str(i)+'.pkl'
        agent.load_model(root+filename1, root+filename2)
    run(agent_list, env, dict)



