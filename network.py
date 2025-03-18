import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, action_size):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(3,3), stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=2)
        self.flatten = nn.Flatten()
        self.fc1  = nn.Linear(512, 128)
        self.fc2a = nn.Linear(128, action_size)
        self.fc2s = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        action_values = self.fc2a(x)
        state_value = self.fc2s(x)[0]
        return action_values, state_value
