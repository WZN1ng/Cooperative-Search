from agent import Agent
from target import Target
from env import EnvSearch
from memory import MemoryReplay
from dqn import DQN

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm

MAP_SIZE = 50
AGENT_NUM = 5
VIEW_RANGE = 6
TARGET_NUM = 9

BATCH_SIZE = 32
NUM_EPISODES = 100
GAMMA = 0.999

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

# collect memory
def collect_memory(agent_list):
    print('collecting memory...')
    memory = agent_list[0].memory
    capacity = agent_list[0].CAPACITY

    while len(memory) < capacity:
        last_obs = []
        total_reward = 0
        for agent in agent_list:    # 重置智能体
            agent.reset()
        # if len(memory) % 1000 == 0:
            # print('memory len: {}/{}'.format(len(memory), Agent.CAPACITY))
        while total_reward > -500:
            done = False
            for i,agent in enumerate(agent_list):
                if len(last_obs) == AGENT_NUM:
                    act = agent.select_action(last_obs[i])
                else:
                    obs_tmp = np.zeros((2*VIEW_RANGE-1, 2*VIEW_RANGE-1))
                    act = agent.select_action(obs_tmp)
                pos = np.array([x/agent.map_size for x in agent.pos])   # 之前的位置

                obs, r, d, _ = env.step(i, agent, act)  # 与环境交互

                # 改变类型
                # obs = obs.astype(np.float32)
                # obs = torch.from_numpy(obs)
                # obs = obs.reshape((-1,1,view,view))
                # pos = pos.astype(np.float32)
                # pos = torch.from_numpy(pos)
                
                # 存储经验
                if len(last_obs) == AGENT_NUM:
                    next_pos = np.array([x/agent.map_size for x in agent.pos])    # 之后的位置
                    # next_pos = next_pos.astype(np.float32)
                    # next_pos = torch.from_numpy(next_pos)
                    if not agent.store_transition(last_obs[i], pos, act, obs, next_pos, r):
                        print('memory length: ', Agent.CAPACITY)
                        return

                if len(last_obs) < AGENT_NUM:
                    last_obs.append(obs)
                else:
                    last_obs[i] = obs
                total_reward += r
                if d:
                    done = True
                    break
            # print('{:.1f}  {}/{}'.format(total_reward, env.target_find, env.target_num))
            if done:
                break

# run
def test(agent_list, env):
    print('test model...')
    env.reset(3, dict)
    for agent in agent_list:
        agent.reset()

    last_obs = []
    total_reward = 0
    done = False
    while True:
        for i,agent in enumerate(agent_list):
            if len(last_obs) == AGENT_NUM:
                act = agent.select_action(last_obs[i])
            else:
                obs_tmp = np.zeros((2*VIEW_RANGE-1, 2*VIEW_RANGE-1))
                act = agent.select_action(obs_tmp)
            # print(i, act)
            obs, r, d, _ = env.step(i, agent, act)
            if len(last_obs) < AGENT_NUM:
                last_obs.append(obs)
            else:
                last_obs[i] = obs
            total_reward += r
            if d:
                done = True
                break 
        print('{:.1f}  {}/{}'.format(total_reward, env.target_find, env.target_num))
        env.render(agent_list, total_reward)
        if done:
            break


# train agent model
def model_train(agent_list, device):
    memory = agent_list[0].memory
    view = 2*agent_list[0].view_range-1
    if len(memory) < BATCH_SIZE:
        return False

    for agent in agent_list:
        transitions = memory.sample(BATCH_SIZE)
        batch = memory.Transition(*zip(*transitions))

        # print(batch.reward, batch.action)
        non_final_mask = torch.tensor(tuple(map(lambda s:s is not None, batch.next_obs)),dtype=torch.bool)
        non_final_next_obs = torch.cat([torch.FloatTensor(s.reshape(1,view,view)) for s in batch.next_obs if s is not None])
        non_final_next_pos = torch.cat([torch.FloatTensor(s) for s in batch.next_pos if s is not None])
        # obs_batch = torch.cat([torch.FloatTensor(s.reshape(1,view,view)) for s in batch.obs])
        # pos_batch = torch.cat([torch.FloatTensor(s) for s in batch.pos])
        # act_batch = [torch.IntTensor(s) for s in batch.action]
        # print(act_batch)
        # reward_batch = [torch.FloatTensor(s) for s in batch.reward]
        
        obs_batch = Variable(torch.from_numpy(np.array(batch.obs).astype(np.float32)).reshape((BATCH_SIZE,1,view,view)))
        pos_batch = Variable(torch.from_numpy(np.array(batch.pos).astype(np.float32)))
        act_batch = Variable(torch.LongTensor(batch.action))
        nobs_batch = Variable(torch.from_numpy(np.array(batch.next_obs).astype(np.float32)).reshape((BATCH_SIZE,1,view,view)))
        npos_batch = Variable(torch.from_numpy(np.array(batch.next_pos).astype(np.float32)))
        reward_batch = Variable(torch.FloatTensor(batch.reward))
        # print(reward_batch.shape, act_batch.shape)
        
        agent.optimizer.zero_grad()
        q0 = agent.model(obs_batch, pos_batch)
        # print(state_action_values)
        q1 = agent.model(nobs_batch, npos_batch).detach()
        # print(torch.max(q1,dim=1)[0].shape)
        q1 = GAMMA*(reward_batch+torch.max(q1,dim=1)[0])
        # print(torch.gather(q0, dim=1, index=act_batch.unsqueeze(1))[:,0].shape)
        loss = agent.loss_function(q1, torch.gather(q0, dim=1, index=act_batch.unsqueeze(1))[:,0])
        loss.backward()
        # for param in agent.model.parameters():
        #     param.grad.data.clamp_(-1,1)
        agent.optimizer.step()

    return True

if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    agent_list = initial_agents(MAP_SIZE, AGENT_NUM, VIEW_RANGE, 1)     # 生成智能体
    dict = {'circle_num':3, 'circle_center':[[4,44],[40,41],[42,3]],
            'circle_radius':[12,12,8],'target_num':[3,4,2],'filename':'targets.txt'}
    env = EnvSearch(MAP_SIZE, TARGET_NUM, 3, dict)      # 生成环境
    # env.save_target('targets2.txt')
    
    collect_memory(agent_list)

    # 开始训练
    print('training...')
    for i_episode in tqdm(range(NUM_EPISODES)):
        # initialize
        env.reset(3, dict)
        for agent in agent_list:
            agent.reset()
            # print(i_episode, agent.pos)
        last_obs = []
        total_reward = 0
        done = False
        while total_reward > -100:
            # 训练
            model_train(agent_list, device)

            # 运行
            for i,agent in enumerate(agent_list):
                if len(last_obs) == AGENT_NUM:
                    act = agent.select_action(last_obs[i])
                else:
                    obs_tmp = np.zeros((2*VIEW_RANGE-1, 2*VIEW_RANGE-1))
                    act = agent.select_action(obs_tmp)
                # print(i, act)
                obs, r, d, _ = env.step(i, agent, act)
                if len(last_obs) < AGENT_NUM:
                    last_obs.append(obs)
                else:
                    last_obs[i] = obs
                total_reward += r
                if d:
                    done = True
                    break 
            # env.render(agent_list, total_reward)
            if done:
                break
        # if done:
        #     print('episode:{}  done:{}  target_found:{}/{}  total_reward:{:.0f}'.format(i_episode, done, env.target_find, TARGET_NUM, total_reward))
    
    # 保存模型参数
    root = './models/'
    for i,agent in enumerate(agent_list):
        file_root = root+'dqn_model_'+str(i)+'.pkl'
        agent.save_model(file_root)
    test(agent_list, env)



