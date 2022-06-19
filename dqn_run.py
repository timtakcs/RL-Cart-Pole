import numpy as np
import gym
import Agent as agnt
import torch
import math

env = gym.make("LunarLander-v2")
print("----------------------------")
print(env.observation_space)
print("----------------------------")
print(env.action_space)
print("----------------------------")

agent = agnt.Agent(env, [0], 8)
agent.net.load_state_dict(torch.load("q_model_lunar.pth"))
agent.net.eval()
agent.net.to("cpu")
state = env.reset()

t_reward = 0

for _ in range(1000):
    action = torch.argmax(agent.net(torch.Tensor(state)))
    new_state, reward, _, _ = env.step(action.item())
    t_reward += reward
    env.render()
    state = new_state

print(t_reward)
    
env.close()