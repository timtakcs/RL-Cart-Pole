import numpy as np
import gym

env = gym.make("CartPole-v0")
print("----------------------------")
print(env.observation_space)
print("----------------------------")
print(env.action_space)
print("----------------------------")

env.reset()
#copied code, change it later

class Agent:
    def __init__(self, evn):
        self.action_size = evn.action_space.n
        print("action space: ", self.action_size)
    
    def getAction(self, state):
        poleAng = state[2]
        if poleAng < 0:
            action = 0
        else:
            action = 1

        return action

agent = Agent(env)
state = env.reset()

for _ in range(1000):
    action = agent.getAction(state)
    state, reward, done, info = env.step(action) 
    env.render()
    
env.close()