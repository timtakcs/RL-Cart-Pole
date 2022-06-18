import numpy as np
import gym
import Agent as agnt
import torch
import math

env = gym.make('CartPole-v1')
print("----------------------------")
print(env.observation_space)
print("----------------------------")
print(env.action_space)
print("----------------------------")

env.render()

agent = agnt.Agent(env, [0], 4)
agent.net.load_state_dict(torch.load("q_model.pth"))
agent.net.eval()
agent.net.to("cpu")
state = env.reset()

for _ in range(1000):
    action = torch.argmax(agent.net(torch.Tensor(state)))
    new_state, _, _, _ = env.step(action.item())
    print(new_state)
    env.render()
    state = new_state
    
env.close()