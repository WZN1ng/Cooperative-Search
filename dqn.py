import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

BATCH_SIZE = 16
LR = 0.005
TARGET_REPLACE_ITER = 100
GAMMA = 0.9
DROP_PROB = 0.5

class Net(nn.Module):
    def __init__(self, view_range):     
        super(Net, self).__init__()
        self.view_range = view_range
        self.size = 2*self.view_range-1 
        self.c1 = nn.Conv2d(1,16,3,1,1)
        self.c2 = nn.Conv2d(16,32,3,1,1)
        self.f1 = nn.Linear(32*self.size*self.size+2,4)
    
    def forward(self, obs, pos):       # 输入单智能体的累积观测阵  numpy.array shape((50,50))
        obs = obs.view(-1,1,self.size,self.size)
        obs = F.relu(self.c1(obs))
        obs = F.relu(self.c2(obs))
        obs = obs.view(obs.size(0),-1)

        pos = pos.view(-1,2)
        # print(obs.shape, pos.shape)
        x = torch.cat([obs,pos], dim=1)
        x = self.f1(x)
        return x
    
class DQN():
    def __init__(self, memory, view_range):
        self.eval_net, self.target_net = Net(view_range), Net(view_range)   # 评价网络与目标网络
        self.learn_step_counter = 0 # 网路参数迭代计数器
        self.memory_counter = 0 # 经验池标记
        self.memory = memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
    
    def learn(self):
        # target net 更新
        if len(self.memory) < BATCH_SIZE:
            return False

        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆
        samples = self.memory.sample(BATCH_SIZE)
        b_o, b_p, b_a, b_o_, b_p_, b_r = self.get_samples(samples)

        q_eval = self.eval_net(b_o, b_p).gather(1, b_a)  # (batch, 1) 选中所有动作的Q值
        q_next = self.target_net(b_o_, b_p_).detach()
        q_target = b_r + GAMMA*torch.unsqueeze(q_next.max(1)[0], 1) #(batch, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return True
    
    def get_samples(self, samples):
        b_o = []
        b_p = []
        b_a = []
        b_o_ = []
        b_p_ = []
        b_r = []
        for sample in samples:
            b_o.append(sample[0])
            b_p.append(sample[1])
            b_a.append(sample[2])
            b_o_.append(sample[3])
            b_p_.append(sample[4])
            b_r.append(sample[5])
        b_o = torch.FloatTensor(b_o)
        b_p = torch.IntTensor(b_p)
        b_a = torch.from_numpy(np.array(b_a).astype(np.int64)).view(-1,1)
        b_o_ = torch.FloatTensor(b_o_)
        b_p_ = torch.IntTensor(b_p_)
        b_r = torch.FloatTensor(b_r).view(-1,1)
        return b_o, b_p, b_a, b_o_, b_p_, b_r

    def get_probs(self, obs, pos):
        return self.eval_net(obs, pos)

    def save_state_dict(self, fileroot1, fileroot2):
        torch.save(self.eval_net.state_dict(), fileroot1)
        torch.save(self.target_net.state_dict(), fileroot2)
    
    def load_state_dict(self, fileroot1, fileroot2):
        self.eval_net.load_state_dict(torch.load(fileroot1))
        self.target_net.load_state_dict(torch.load(fileroot2))

if __name__ == "__main__":
    n = Net(6)
    view = 11
    obs = torch.from_numpy(np.ones((view,view)).astype(np.float32))
    pos = torch.from_numpy(np.array([[24,25]]).astype(np.int32))
    print(n(obs,pos).detach().numpy())