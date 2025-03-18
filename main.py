from agent import Agent
from env_batch import EnvBatch
from train import train
from evaluate import evaluate
from visualize import show_video_of_model
import numpy as np

number_actions = 14
agent = Agent(number_actions)
env_batch = EnvBatch(10)

train(agent, env_batch)

print("Average agent reward: ", np.mean(evaluate(agent, env_batch.envs[0], n_episodes=10)))

show_video_of_model(agent, "KungFuMasterDeterministic-v4")
