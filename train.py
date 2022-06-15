import numpy as np
import gym
import Agent as agnt
import math
import torch

env = gym.make("CartPole-v1")
print("----------------------------")
print(env.observation_space)
print("----------------------------")
print(env.action_space)
print("----------------------------")

env.reset()

np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])
observation = [30, 30, 50, 50]

#init parameters
learning_rate = 0.1
explore_rate = 1
max_episodes = 30000
prior_reward = 0
total_reward = 0

discount = 0.95
decay_value = 0.99995

#bucket sizes are (30, 30, 50, 50)
#since q learning assumes discrete states, we need to discretize the continuous cartpole states
def discretize(state):
    discrete_state = state/np_array_win_size + np.array([15, 10, 1, 10])
    return tuple(discrete_state.astype(np.int))


agent = agnt.Agent(env, observation)

#main algorithm loop
for episode in range(max_episodes):
    disc_state = discretize(env.reset())
    done = False
    ep_reward = 0

    while not done:
        action = agent.getAction(disc_state, explore_rate, env)
        new_state, reward, done, info = env.step(action)
        ep_reward += reward
        new_disc_state = discretize(new_state)

        #so that i dont have to render all of the episodes
        if episode % 1000 == 0:
            env.render()

        #update the Q table
        if not done:
            opt_est = np.max(agent.q_table[new_disc_state])
            current = agent.q_table[disc_state + (action,)]
            agent.q_table[disc_state + (action,)] = (1 - learning_rate) * current + learning_rate * (reward + discount * opt_est)

        disc_state = new_disc_state

        if explore_rate > 0.05:
            if ep_reward > prior_reward and episode > 1000:
                print(f'{(30000 - episode)} batches left')
                explore_rate = math.pow(decay_value, episode - 1000)

            print("Explore rate value: ", explore_rate)

        prior_reward = ep_reward
        total_reward += ep_reward

    if episode % 1000 == 0:
        mean = total_reward / 1000
        print("mean reward for the last 10000 episodes was:", mean)
        total_reward = 0

np.save('q_table', agent.q_table)
state = discretize(env.reset())

env.close()

