import torch
import torch.nn.functional as F
import numpy as np
from network import Network

learning_rate = 1e-4
discount_factor = 0.99

class Agent():
    def __init__(self, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.network = Network(action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    def act(self, state):
        if state.ndim == 3:
            state = [state]
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        action_values, _ = self.network.forward(state)
        policy = F.softmax(action_values, dim=-1)
        return np.array([np.random.choice(len(p), p=p) for p in policy.detach().cpu().numpy()])

    def step(self, state, action, reward, next_state, done):
        batch_size = state.shape[0]
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.bool, device=self.device).to(dtype=torch.float32)

        action_values, state_value = self.network(state)
        _, next_state_value = self.network(next_state)
        target_state_value = reward.unsqueeze(1) + discount_factor * next_state_value * (1 - done.unsqueeze(1))
        advantage = target_state_value - state_value

        probs = F.softmax(action_values, dim=-1)
        logprobs = F.log_softmax(action_values, dim=-1)
        entropy = -torch.sum(probs * logprobs, axis=-1)

        batch_idx = np.arange(batch_size)
        action = torch.tensor(action, dtype=torch.long, device=self.device)
        logp_actions = logprobs.gather(1, action.unsqueeze(1)).squeeze(1)

        actor_loss = -(logp_actions * advantage.detach()).mean() - 0.001 * entropy.mean()
        critic_loss = F.mse_loss(target_state_value.detach(), state_value)
        total_loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
