import torch.nn as nn
import torch.nn.functional as f 

class OffPGCritic(nn.Module):
    def __init__(self, input_shape, args):
        super(OffPGCritic, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.offpg_hidden_dim)
        self.fc2 = nn.Linear(args.offpg_hidden_dim, args.offpg_hidden_dim)
        self.fc_v = nn.Linear(args.offpg_hidden_dim, 1)
        self.fc3 = nn.Linear(args.offpg_hidden_dim, args.n_actions)

    def forward(self, inputs):  # inputs (state, obs, ...)
        x = f.relu(self.fc1(inputs))
        x = f.relu(self.fc2(x))
        v = self.fc_v(x)
        a = self.fc3(x)
        q = a + v
        return q