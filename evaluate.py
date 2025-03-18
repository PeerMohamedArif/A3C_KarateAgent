import numpy as np

def evaluate(agent, env, n_episodes=1):
    episodes_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()

        if len(state.shape) == 3 and state.shape[0] != 4:
            state = np.expand_dims(state, axis=0)
            state = np.repeat(state, 4, axis=0)

        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)[0]
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            if len(next_state.shape) == 3 and next_state.shape[0] != 4:
                next_state = np.expand_dims(next_state, axis=0)
                next_state = np.repeat(next_state, 4, axis=0)

            next_state_stack = np.roll(state, shift=-1, axis=0)
            next_state_stack[-1] = np.squeeze(next_state[0])
            state = next_state_stack

        episodes_rewards.append(total_reward)

    return episodes_rewards
