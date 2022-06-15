import numpy as np
import gym
import Agent as agnt
import torch

env = gym.make("CartPole-v1")
print("----------------------------")
print(env.observation_space)
print("----------------------------")
print(env.action_space)
print("----------------------------")