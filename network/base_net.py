import torch
import torch.nn as nn
import torch.nn.functional as f 

class RNN(nn.Module):
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args
        self.input_shape = input_shape
        if args.conv:
            self.conv_size = int((args.map_size-args.kernel_size_1)/args.stride_1 + 1)
            self.conv = nn.Sequential(
                nn.Conv2d(1, args.dim_1, args.kernel_size_1, args.stride_1), # (1, 50, 50) --> (4, 24, 24)
                nn.ReLU(),
                nn.Conv2d(args.dim_1, args.dim_2, args.kernel_size_2, args.stride_2, args.padding_2), # (4, 24, 24) --> (8, 24, 24)
                nn.ReLU()
                # 
            )
            self.linear = nn.Linear(args.dim_2*self.conv_size**2, args.conv_out_dim)
            # input_shape += args.conv_out_dim
        # print('input shape: ', input_shape)
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim, args.n_actions)
        )

    def forward(self, obs, hidden_state):
        if self.args.conv:
            prob = obs[:, :self.args.map_size**2].reshape(-1 ,1 ,self.args.map_size, self.args.map_size)
            sa = obs[:, self.args.map_size**2:]
            prob_conv = self.conv(prob).reshape(-1, self.args.dim_2*self.conv_size**2)
            # print(prob_conv.shape)
            prob_conv = self.linear(prob_conv)
            # print(prob_conv.shape)
            # print('prob sa ', prob_conv.shape, sa.shape)
            obs = torch.cat([prob_conv, sa], 1)
            # print(self.conv_size)
            # print('obs ', obs.shape, self.input_shape)
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

    