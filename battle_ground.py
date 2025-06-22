import gym
env = gym.make("PongNoFrameskip-v4")
obs = env.reset()
env.render()
input("Press any key...")
env.close()