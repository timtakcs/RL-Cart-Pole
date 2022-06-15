import random
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(4, 24)
        self.layer2 = nn.Linear(24, 24)
        self.layer3 = nn.Linear(24, 24)
        self.layer4 = nn.Linear(24, 2)

        self.apply(self.initialize_weights)

    def forward(self, data):
        data = f.relu(self.layer1(data))
        data = f.relu(self.layer2(data))
        data = f.relu(self.layer3(data))

        return f.linear(self.layer4(data))

    def initialize_weights(layer):
        nn.init.kaiming_uniform_(layer.weight.data)
        nn.init.constant_(layer.bias.data, 0)

class Agent:
    def __init__(self, env, observation, net=None):
        self.observation_size = (1, 1, 6, 3)
        self.action_size = env.action_space.n
        self.q_table = np.random.uniform(low=0, high=1, size=(observation + [self.action_size]))
        self.q_table.shape
        self.replay_memory = []
        self.net = Net()
        self.target_net = Net()
           
    def getAction(self, state, exploration_rate, env):
        if random.random() < exploration_rate:
            action = env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state])

        return action    

    def train(self, state, e, lr):
        pass

    def get_q_state(self):
        pass