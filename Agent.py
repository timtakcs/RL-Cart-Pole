import random
import numpy as np

class Agent:
    def __init__(self, env, observation):
        self.observation_size = (1, 1, 6, 3)
        self.action_size = env.action_space.n
        self.q_table = np.random.uniform(low=0, high=1, size=(observation + [self.action_size]))
        self.q_table.shape

    
    def getAction(self, state, exploration_rate, env):
        if random.random() < exploration_rate:
            action = env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state])

        return action