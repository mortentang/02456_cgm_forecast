import os

import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import add, nn, tensor, device

device = torch.device('cuda:0')


# device = 'cpu'

# @ray.remote
class DilatedNet(nn.Module):
    def __init__(self, n_steps_past=16, num_inputs=4,
                 dilations=[1, 1, 2, 4, 8],
                 h1=2,
                 h2=3,
                 h3=3,
                 h4=3):  # [32,32,32,64,64]):

        """
        :param num_inputs: int, number of input variables
        :param h1: int, size of first three hidden layers
        :param h2: int, size of last two hidden layers
        :param dilations: int, dilation value
        :param hidden_units:
        """

        super(DilatedNet, self).__init__()
        self.file_name = os.path.basename(__file__)
        self.hidden_units = [h1, h2, h3, h4, h2]
        self.dilations = dilations

        self.num_inputs = num_inputs
        self.receptive_field = sum(dilations) + 1

        self.input_width = n_steps_past  # n steps past

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(self.num_inputs, self.hidden_units[0], kernel_size=3)
        self.batchnorm_c1 = nn.BatchNorm1d(self.hidden_units[0])
        self.conv2 = nn.Conv1d(self.hidden_units[0], self.hidden_units[1], kernel_size=3)
        self.batchnorm_c2 = nn.BatchNorm1d(self.hidden_units[1])
        self.conv3 = nn.Conv1d(self.hidden_units[1], self.hidden_units[2], kernel_size=3)
        self.batchnorm_c3 = nn.BatchNorm1d(self.hidden_units[2])
        self.conv4 = nn.Conv1d(self.hidden_units[2], 9, kernel_size=3)
        # self.conv5 = nn.Conv1d(self.hidden_units[3], 7, kernel_size=3)
        self.maxpool1d = nn.MaxPool1d(1, 1)

        self.lstm = nn.LSTM(input_size=9,
                            hidden_size=32,
                            num_layers=1,
                            batch_first=True)

        self.l_1 = nn.Linear(in_features=32,
                             out_features=64,
                             bias=False)

        self.l_2 = nn.Linear(in_features=64,
                             out_features=32,
                             bias=False)

        # Output layer
        self.l_out = nn.Linear(in_features=32,
                               out_features=1,
                               bias=False)

        self.drop = nn.Dropout()
        self.drop = self.drop.to(self.device)

        self.relu = self.relu.to(self.device)

        self.conv1 = self.conv1.to(self.device)
        self.conv2 = self.conv2.to(self.device)
        self.conv3 = self.conv3.to(self.device)
        self.conv4 = self.conv4.to(self.device)
        # self.conv5 = self.conv5.to(self.device)
        self.maxpool1d = self.maxpool1d.to(self.device)
        self.lstm = self.lstm.to(self.device)
        self.l_1 = self.l_1.to(self.device)
        self.l_2 = self.l_2.to(self.device)
        self.l_out = self.l_out.to(self.device)

    def forward(self, x):
        """
        :param x: Pytorch Variable, batch_size x n_stocks x T
        :return:
        """

        # First layer
        current_width = x.shape[2]
        pad = max(self.receptive_field - current_width, 0)
        input_pad = nn.functional.pad(x, [pad, 0], "constant", 0)
        x = self.conv1(input_pad)
        x = self.relu(x)
        x = self.maxpool1d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool1d(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool1d(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool1d(x)

        h0 = torch.zeros(1, x.size(0), 32).to(self.device)
        c0 = torch.zeros(1, x.size(0), 32).to(self.device)

        x, _ = self.lstm(x, (h0, c0))
        x = F.dropout(x, training=self.training)
        # Remove redundant dimensions
        out = x[:, -1, :]
        x = self.l_1(out)
        x = F.dropout(x, training=self.training)
        x = self.l_2(x)
        x = F.dropout(x, training=self.training)
        out_final = self.l_out(x)
        return out_final
