from agent import Agent
from target import Target
from env import EnvSearch

import numpy as np


MAP_SIZE = 100
AGENT_NUM = 10
VIEW_RANGE = 4
TARGET_NUM = 30

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
                temp_agent = Agent([i,j], view_range)
                agent_list.append(temp_agent)
    return agent_list


if __name__ == '__main__':
    agent_list = initial_agents(MAP_SIZE, AGENT_NUM, VIEW_RANGE, 1)     # 生成智能体
    dict = {'circle_num':3, 'circle_center':[[16,60],[70,70],[66,20]],
            'circle_radius':[12,12,8],'target_num':[10,15,5]}
    env = EnvSearch(MAP_SIZE, TARGET_NUM, 2, dict)      # 生成环境
    total_reward = 0

    while True:
        done = False
        for agent in agent_list:
            act = np.random.randint(0, 4, size=1)
            obs, r, d, _ = env.step(agent, act)
            total_reward += r
            if d:
                done = True
                break 
        print('{:.1f}  {}/{}'.format(total_reward, env.target_find, env.target_num))
        if done:
            break
        env.render(agent_list)

