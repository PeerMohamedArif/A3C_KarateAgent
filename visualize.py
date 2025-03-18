import gymnasium as gym
import cv2
import numpy as np

def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False

    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = cv2.resize(state, (42, 42)) / 255.
    state = np.expand_dims(state, axis=0)
    state = np.repeat(state, 4, axis=0)

    frame_shape = env.render().shape
    height, width, layers = frame_shape
    video_writer = cv2.VideoWriter(
        'karate_video2.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        (width, height)
    )

    while not done:
        frame = env.render()
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        action = agent.act(state)[0]
        next_state, reward, done, _, _ = env.step(action)

        next_state = cv2.cvtColor(next_state, cv2.COLOR_RGB2GRAY)
        next_state = cv2.resize(next_state, (42, 42)) / 255.
        next_state = np.expand_dims(next_state, axis=0)

        next_state_stack = np.roll(state, shift=-1, axis=0)
        next_state_stack[-1] = np.squeeze(next_state)
        state = next_state_stack

    env.close()
    video_writer.release()
