import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(4)
        # self.conv2 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(16)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(32)
        self.linear = nn.Linear(2, 16)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc = nn.Linear(16*16, 4)
        
    def forward(self, x):
        x = self.linear(x).reshape(-1,1,16)
        x = F.relu(self.bn1(self.conv1(x))).reshape(-1,256)  # 一层卷积
        # print(x.shape)
        # obs = F.relu(self.bn2(self.conv2(obs)))  # 两层卷积
        # obs = F.relu(self.bn3(self.conv3(obs)))  # 三层卷积
        return self.fc(x)
        # if obs.shape[0] == 1:
        #     return self.fc(torch.cat((obs.view(-1), pos),0))  # 全连接层
        # else:
        #     l = obs.shape[0] 
        #     # print(obs.shape, obs.view(l,-1).shape, pos.shape, torch.cat((obs.view(l,-1), pos),1).shape)
        #     return self.fc(torch.cat((obs.view(l,-1), pos),1))