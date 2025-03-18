import tqdm
from env_batch import EnvBatch
from agent import Agent

def train(agent, env_batch, num_iterations=3001):
    batch_states = env_batch.reset()
    with tqdm.trange(0, num_iterations) as progress_bar:
        for i in progress_bar:
            batch_actions = agent.act(batch_states)
            batch_next_states, batch_rewards, batch_dones = env_batch.step(batch_actions)
            batch_rewards *= 0.01
            agent.step(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)
            batch_states = batch_next_states
