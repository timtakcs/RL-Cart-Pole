import numpy as np
import gym
import Agent as agnt
import torch
import math
import os

env = gym.make("CartPole-v1")
print("----------------------------")
print(env.observation_space)
print("----------------------------")
print(env.action_space)
print("----------------------------")

#init the agent
agent = agnt.Agent(env, [0])

#hyperparameters
max_episodes = 10000
test_episodes = 1000
discount = 0.85
decay = 0.9995
prior_reward = 0
batch_size = 10

epsilon = 1
min_epsilon = 0.1

c = 0

for episodes in range(max_episodes):
    done = False
    ep_reward = 0
    state = env.reset()

    while not done:
        for steps in range(batch_size):
            action = agent.get_action(state, epsilon, env)
            new_state, reward, done, info = env.step(action)
            agent.replay_memory.append((state, action, reward, new_state))

        agent.train(discount, episodes)

        if epsilon > 0.05:
            if ep_reward > prior_reward and episodes > 100:
                print(f'{(1000 - episodes)} episodes left')
                explore_rate = math.pow(decay, episodes - 1000)

    if episodes % 100 == 0:
        print(f'{max_episodes - episodes} left')

torch.save(agent.net.state_dict(), "q_model.pth")

# total_up = 0

# state = env.reset()

# for _ in range(test_episodes):
#     action = torch.argmax(agent.get_action(state, 0, env))
#     new_state, _, _, _ = env.step(action.item())
#     print(new_state)
#     env.render()
#     state = new_state
    
env.close()



# state = (14, 8, -41, -2)
# action = 0
# reward = 1
# next_state = (14, 8, -44, 0)

# agent.replay_memory.append((state, action, reward, next_state))
# agent.train(discount, 1)

