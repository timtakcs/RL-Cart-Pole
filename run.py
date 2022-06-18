import Agent as agnt
import gym
import numpy as np

env = gym.make("CartPole-v1")
data = np.load('q_table.npy')

agent = agnt.Agent(env, [0])
agent.q_table = data

#sorry for redundancies
np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1])

print("sDD", np.arange(10, dtype=np.int32))

#sorry again
def discretize(state):
    discrete_state = state/np_array_win_size + np.array([15, 10, 1, 10])
    return tuple(discrete_state.astype(np.int))

state = discretize(env.reset())

for _ in range(1000):
    print(state)
    action = np.argmax(agent.q_table[state])
    print(action)
    new_state, _, _, _ = env.step(action)
    env.render()

    state = discretize(new_state)