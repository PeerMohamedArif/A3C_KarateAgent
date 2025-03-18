import cv2
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

class PreprocessAtari(ObservationWrapper):
    def __init__(self, env, height=42, width=42, crop=lambda img: img, dim_order='pytorch', color=False, n_frames=4):
        super(PreprocessAtari, self).__init__(env)
        self.img_size = (height, width)
        self.crop = crop
        self.dim_order = dim_order
        self.color = color
        self.frame_stack = n_frames
        n_channels = 3 * n_frames if color else n_frames
        obs_shape = {'tensorflow': (height, width, n_channels), 'pytorch': (n_channels, height, width)}[dim_order]
        self.observation_space = Box(0.0, 1.0, obs_shape, dtype=np.float32)
        self.frames = np.zeros(obs_shape, dtype=np.float32)

    def reset(self):
        self.frames = np.zeros_like(self.frames)
        obs, info = self.env.reset()
        self.update_buffer(obs)
        return self.frames, info

    def observation(self, img):
        img = self.crop(img)
        img = cv2.resize(img, self.img_size)
        if not self.color:
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype('float32') / 255.
        self.frames = np.roll(self.frames, shift=-1 if not self.color else -3, axis=0)
        self.frames[-1 if not self.color else -3:] = img
        return self.frames

    def update_buffer(self, obs):
        self.frames = self.observation(obs)
