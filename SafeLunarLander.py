from typing import Optional

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


# Source - https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/box2d/lunar_lander.py
class SafeLunarLander(gym.Env, *args, **kwargs):
    def __init__(self):
        self.env = gym.make('LunarLander-v2')  # Initialize the original Lunar Lander
        self.path = []  # To store the path points
        super().__init__()

    def step(self, action):
        observation, reward, done, _, info = super().step(action=action)
        x, y = observation[0], observation[1]
        self.path.append((x, y))  # Append current position to path
        if done:
            reward += self.score_landing_path(self.path, x, y)  # Add path score to final reward
        return observation, reward, done, info

    def score_landing_path(self, points, x0, y0, a=0.2):
        inside_parabola = 0
        total_points = len(points)
        for x, y in points:
            vertical_distance = y0 - y
            max_horizontal_distance = np.sqrt(a * vertical_distance)
            if abs(x - x0) <= max_horizontal_distance:
                inside_parabola += 1
        score = (inside_parabola / total_points) * 100
        return score

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.path = []
        super().reset(seed=seed)
