import gym
import minihack
env1 = gym.make("MiniHack-MultiRoom-N6-v0")
env1.reset()

import mybabyai
env2 = gym.make("MyBabyAI-GoToS8R2D1-v0")
env2.reset()

#atari
env3 = gym.make("ALE/Adventure-v5")
env3.reset()

import babyai
env3 = gym.make("BabyAI-GoToRedBall-v0")
env3.reset()

import gym_minigrid
env4 = gym.make("MiniGrid-FourRooms-v0")
env4.reset()

print("oookkkk")