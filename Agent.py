import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as opt
import math

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(4, 24)
        self.layer2 = nn.Linear(24, 24)
        self.layer3 = nn.Linear(24, 24)
        self.layer4 = nn.Linear(24, 2)

        self.initialize_weights()

    def forward(self, data):
        data = f.relu(self.layer1(data))
        data = f.relu(self.layer2(data))
        data = f.relu(self.layer3(data))

        return self.layer4(data)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)

class Agent:
    def __init__(self, env, observation, net=None):
        self.observation_size = (1, 1, 6, 3)
        self.action_size = env.action_space.n
        self.q_table = np.random.uniform(low=0, high=1, size=(observation + [self.action_size]))
        self.q_table.shape
        self.replay_memory = []
        self.net = Net()
        self.target_net = Net()
        self.optimizer = opt.Adam(self.net.parameters(), lr=1e-3)
           
    def get_action(self, state, epsilon, env):
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(self.net(torch.Tensor(state)))

        return action   

    def get_action_q(self, state, exploration_rate, env):
        if random.random() < exploration_rate:
            action = env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state])

        return action    

    def train(self, discount, eps):
        replay_memory = self.shuffle(self.replay_memory)
        x = []
        y = []

        for i in range(len(replay_memory)):
            current_exp = replay_memory[i]
            state, action, reward, next_state = current_exp[0], current_exp[1], current_exp[2], current_exp[3]
            out = self.net(torch.Tensor(state))
            q_value = out[action]

            target_q_value = reward + discount * torch.max(self.target_net(torch.Tensor(next_state)))
            
            x.append(q_value)
            y.append(target_q_value)
            
        loss = f.mse_loss(torch.Tensor(x), torch.Tensor(y))
        loss.requires_grad = True
        print("loss", loss)
        self.net.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.replay_memory = []

        if eps % 100 == 0:
            self.target_net.load_state_dict(self.net.state_dict())

    # def mse(self, loss_calc, batch_size):
    #     total = 0

    #     print(loss_calc[0])

    #     for _ in range(batch_size):
    #         total += math.pow(loss_calc[0][0] - loss_calc[1][0], 2)

    #     return total/batch_size

    def shuffle(self, array):
        for i in range(len(array)):
            swap_idx = random.randrange(i, len(array))
            array[i], array[swap_idx] = array[swap_idx], array[i]
        return array