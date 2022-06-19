from collections import deque
import random
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as opt
import math

class Net(nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_size, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 32)
        self.layer4 = nn.Linear(32, 4)
        self.initialize_weights()
        self.optimizer = opt.Adam(self.parameters(), lr=1e-3)
        self.loss = nn.MSELoss()

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

class ConvNet(nn.Module):
    def __init__(self, input_size) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.layer1 = nn.Linear(64, 128)
        self.layer2 = nn.Linear(128, 256)
        self.output_layer = nn.Linear(256, 3)

    def forward(self, data):
        data = f.relu(self.conv1(data))
        data = f.max_pool2d(data)
        data = f.relu(self.conv2(data))
        data = f.max_pool2d(data)
        data = nn.Flatten(data)
        data = f.relu(self.layer1(data))
        data = f.relu(self.layer2(data))

        return f.softmax(self.output_layer(data))

class Agent:
    def __init__(self, env, observation, input_size):
        # self.observation_size = (1, 1, 6, 3)
        # self.action_size = env.action_space.n
        # self.q_table = np.random.uniform(low=0, high=1, size=(observation + [self.action_size]))
        # self.q_table.shape
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = Net(input_size).to(self.device)
        self.target_net = Net(input_size).to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.memory_cnt = 0
        self.memory_size = 1000

        self.state_memory = np.zeros((self.memory_size, input_size), dtype=np.float32)
        self.next_state_memory = np.zeros((self.memory_size, input_size), dtype=np.float32)
        self.action_memory = np.zeros(self.memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.int32)
        self.done_memory = np.zeros(self.memory_size, dtype=bool)
           
    def get_action(self, state, epsilon, env):
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(self.net(torch.Tensor(state).to(self.device))).item()

        return action   

    def get_action_q(self, state, epsilon, env):
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state])
            print(action)

        return action    

    # def train(self, discount, eps, batch_size):
    #     if self.memory_cnt < batch_size:
    #         return

    #     self.net.optimizer.zero_grad()

    #     # max_memory = min(self.memory_cnt, self.memory_size)
    #     # batch = np.random.choice(max_memory, batch_size, replace=False)
    #     # batch = np.array(self.replay_memory, dtype=np.object_)[batch]
    #     batch = self.shuffle(self.replay_memory)

    #     for i in range(batch_size):
    #         current_exp = batch[i]
    #         state = current_exp[0]
    #         action = current_exp[1]
    #         reward = current_exp[2]
    #         next_state = current_exp[3]
    #         done = current_exp[4]
    #         out = self.net(torch.Tensor(state))
    #         print(out)
    #         q_value = out[action]
            
    #         target_q_value = reward + discount * torch.max(self.target_net(torch.Tensor(next_state)))

    #         loss = self.net.loss(torch.Tensor(q_value), torch.Tensor(target_q_value))
    #         loss.backward()
    #         self.net.optimizer.step()

    #     if eps % 100 == 0:
    #         self.target_net.load_state_dict(self.net.state_dict())

    def train(self, discount, frame, batch_size):
        if self.memory_cnt < batch_size:
            return

        self.net.optimizer.zero_grad()

        max_memory = min(self.memory_cnt, self.memory_size)
        batch = np.random.choice(max_memory, batch_size, replace=False)

        batch_index = np.arange(batch_size, dtype=np.int32)
        
        state = torch.Tensor(self.state_memory[batch]).to(self.device)
        next_state = torch.Tensor(self.next_state_memory[batch]).to(self.device)
        reward = torch.Tensor(self.reward_memory[batch]).to(self.device)
        done = torch.ByteTensor(self.done_memory[batch]).to(self.device)

        action = self.action_memory[batch]

        q_value = self.net.forward(state)[batch_index, action]
        q_next = torch.max(self.target_net.forward(next_state), dim=1)[0]
        q_next[done] = 0.0
        q_next.detach()
        
        q_target = reward + discount * q_next

        loss = self.net.loss(q_value, q_target).to(self.device)
        loss.backward()
        self.net.optimizer.step()

        if frame % 1000 == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        # if self.epsilon > self.min_epsilon:
        #     self.decay()

    # def decay(self):
    #     self.epsilon = (max(self.min_epsilon, min(1.0, 1.0 - math.log10((self.epsilon + 1)/25))))

    def store(self, state, action, reward, next_state, done):
        index = self.memory_cnt % self.memory_size
        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.done_memory[index] = done
        self.memory_cnt += 1

    # def shuffle(self, array):
    #     for i in range(len(array)):
    #         swap_idx = random.randrange(i, len(array))
    #         array[i], array[swap_idx] = array[swap_idx], array[i]
    #     return array