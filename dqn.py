import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, view_range):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        vtmp = 2*view_range-1
        self.fc = nn.Linear(32*vtmp*vtmp+2, 4)
        
    def forward(self, obs, pos):
        obs = F.relu(self.bn1(self.conv1(obs)))  # 一层卷积
        obs = F.relu(self.bn2(self.conv2(obs)))  # 两层卷积
        obs = F.relu(self.bn3(self.conv3(obs)))  # 三层卷积
       
        if obs.shape[0] == 1:
            return self.fc(torch.cat((obs.view(-1), pos),0))  # 全连接层
        else:
            l = obs.shape[0] 
            # print(obs.shape, obs.view(l,-1).shape, pos.shape, torch.cat((obs.view(l,-1), pos),1).shape)
            return self.fc(torch.cat((obs.view(l,-1), pos),1))