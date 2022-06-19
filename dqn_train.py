import numpy as np
import gym
import Agent as agnt
import torch
import matplotlib.pyplot as plt
import os

env = gym.make("LunarLander-v2")
print("----------------------------")
print(env.observation_space)
print("----------------------------")
print(env.action_space)
print("----------------------------")

#init the agent
agent = agnt.Agent(env, [0], 8)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#hyperparameters
max_episodes = 500
test_episodes = 1000
discount = 0.8
decay = 0.995
batch_size = 32

epsilon = 1
min_epsilon = 0.001

frame = 0

scores = []
eps_for_plot = []

mean = []

for episodes in range(max_episodes):
    done = False
    ep_reward = 0
    state = env.reset()

    print(max_episodes - episodes)

    while not done:
        frame += 1
        action = agent.get_action(state, epsilon, env)
        new_state, reward, done, info = env.step(action)
        agent.store(state, action, reward, new_state, done)
        
        agent.train(discount, frame, batch_size)

        state = new_state
        ep_reward += reward

        if episodes > 100:
            epsilon = max(epsilon * decay, min_epsilon)

    scores.append(ep_reward)
    eps_for_plot.append(episodes)
    
    ep_reward = 0

    if episodes % 100 == 0:
        print(f'{max_episodes - episodes} left')
        print("epsilon:", epsilon)
        if episodes >= 100:
            mean.append(np.mean(scores[-100:]))

plt.plot(eps_for_plot, scores)
plt.show()

print(mean)

torch.save(agent.net.state_dict(), "q_model_lunar.pth")
    
env.close()


