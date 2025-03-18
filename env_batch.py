import numpy as np
import gymnasium as gym
import ale_py  # ⚠️ Keeping this import as requested!
from preprocess_atari import PreprocessAtari

def make_env():
    env = gym.make("KungFuMasterDeterministic-v4", render_mode='rgb_array')
    return PreprocessAtari(env, height=42, width=42)

class EnvBatch:
    def __init__(self, n_envs=10):
        self.envs = [make_env() for _ in range(n_envs)]

    def reset(self):
        return np.array([env.reset()[0] for env in self.envs])

    def step(self, actions):
        next_states, rewards, dones, _, _ = map(np.array, zip(*[env.step(a) for env, a in zip(self.envs, actions)]))
        for i in range(len(self.envs)):
            if dones[i]:
                next_states[i] = self.envs[i].reset()[0]
        return next_states, rewards, dones
