import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

BATCH_SIZE = 16
LR = 0.005
TARGET_REPLACE_ITER = 100
GAMMA = 0.9

class Net(nn.Module):
    def __init__(self):     
        super(Net, self).__init__()
        self.c1 = nn.Conv2d(1,16,5,2,1)
        self.f1 = nn.Linear(16*24*24,4)
    
    def forward(self, x):       # 输入单智能体的累积观测阵  numpy.array shape((50,50))
        # print('1',x.shape)
        x = x.view(-1,1,50,50)
        x = F.relu(self.c1(x))
        # print('2',x.shape)
        x = x.view(-1,16*24*24)
        x = self.f1(x)
        # print('3',x.shape)
        return x
    
class DQN():
    def __init__(self, memory):
        self.eval_net, self.target_net = Net(), Net()   # 评价网络与目标网络
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
        b_s = []
        b_a = []
        b_r = []
        b_s_ = []
        for sample in samples:
            b_s.append(sample[0])
            b_a.append(sample[1])
            b_s_.append(sample[2])
            b_r.append(sample[3])
        b_s = torch.FloatTensor(b_s)
        b_a = torch.from_numpy(np.array(b_a).astype(np.int64)).view(-1,1)
        b_r = torch.FloatTensor(b_r).view(-1,1)
        b_s_ = torch.FloatTensor(b_s_)
        # print(b_s.shape, b_a.shape, b_r.shape, b_s_.shape)
        # print(self.eval_net(b_s).shape, b_a.shape)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # (batch, 1) 选中所有动作的Q值
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA*torch.unsqueeze(q_next.max(1)[0], 1) #(batch, 1)
        # print(q_eval.shape, q_target.shape, b_r.shape, torch.unsqueeze(q_next.max(1)[0], 1).shape)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return True

    def get_probs(self, x):
        return self.eval_net(x)

    def save_state_dict(self, fileroot1, fileroot2):
        torch.save(self.eval_net.state_dict(), fileroot1)
        torch.save(self.target_net.state_dict(), fileroot2)
    
    def load_state_dict(self, fileroot1, fileroot2):
        self.eval_net.load_state_dict(torch.load(fileroot1))
        self.target_net.load_state_dict(torch.load(fileroot2))

# if __name__ == "__main__":