from typing import Any

from gymnasium import Wrapper
import numpy as np
from gymnasium.core import WrapperObsType

import safePara


class SafeLunarLanderWrapper(Wrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env)
        self.path = []
        self.safe_penalties = []
        self.debug = kwargs.get('debug', False)

        self.a = 1  # Coefficient for x^2
        self.b = 0  # Coefficient for x
        self.c = 0.04  # Constant term
        self.max_penalty = -10  # Maximum penalty to apply

    def score_landing_path(self, x0, y0):

        # The flags on the map are between -0.2 to 0.2
        if x0 > 0.2 or x0 < -0.2:
            print("EXIT")
            return -100

        inside_parabola = 0
        a = 0.2
        total_points = len(self.path)

        for x, y in self.path:
            vertical_distance = abs(y0 - y)
            max_horizontal_distance = np.sqrt(a * vertical_distance)
            if abs(x - x0) <= max_horizontal_distance:
                inside_parabola += 1

        score = (inside_parabola / total_points) * 100

        print("SCORING", inside_parabola, total_points, score)
        return score

    def descent_path_landing_score(self, x, y):
        # The flags on the map are between -0.2 to 0.2
        inside_parabola = 0
        a = 0.2

        vertical_distance = abs(y)

        max_horizontal_distance = np.sqrt(a * vertical_distance)

        #print("hri", y, abs(x), max_horizontal_distance/2)

        if abs(x) <= max_horizontal_distance:
            return 0

        else:
            return - ((abs(x) - max_horizontal_distance) / 4*(1.4-y))


    def y_based(self, x, y):
        y_desired = self.a * x ** 2 + self.b * x + self.c

        # Calculate altitude scaling factor
        altitude_scaling_factor = 1 - y / self.env.observation_space.high[1]  # Normalize using the max y from space

        # Reward shaping: only penalize if y is below y_desired
        if y < y_desired:
            deviation = y_desired - y
            # Calculate a scaled penalty that increases as the lander descends
            penalty = -np.sqrt(deviation) * altitude_scaling_factor
            # Ensure the penalty does not exceed maximum allowed penalty
            shaped_reward = max(self.max_penalty, penalty)
        else:
            shaped_reward = 0

        return shaped_reward


    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        """Uses the :meth:`reset` of the :attr:`env` that can be overwritten to change the returned data."""
        self.path = []
        self.safe_penalties = []
        return self.env.reset(seed=seed, options=options)

    # def step(self, action):
    #     observation, reward, done, _, info = self.env.step(action)
    #     x, y = observation[0], observation[1]
    #     self.path.append((x, y))  # Append current position to path
    #     if done:
    #         land_score = self.score_landing_path(x, y)  # Add path score to final reward
    #         if self.debug:
    #             print("Safe land score", reward, land_score)
    #             safePara.plot_landing_path(self.path, 0.167, x, y, land_score)
    #         reward += land_score
    #     return observation, reward, done, _, info

    def step(self, action):
        observation, reward, done, _, info = self.env.step(action)
        x, y = observation[0], observation[1]
        self.safe_penalties.append(self.y_based(x, y))
        self.path.append((x, y))
        #print(y, x, self.safe_penalties[-1], reward)
        reward += self.safe_penalties[-1]
        info["safety"] = 0
        if done:
            info["safety"] = self.score_landing_path(x, y)
        if done and self.debug:
            # print(self.safe_penalties)
            safePara.plot_landing_path(self.path, 0.2, x, y, sum(self.safe_penalties))
        return observation, reward, done, _, info
