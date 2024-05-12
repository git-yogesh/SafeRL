from typing import Any
from gymnasium import Wrapper
from gymnasium.core import WrapperObsType
import numpy as np
import scripts.SafePara as SafePara


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
        self.reduced_engine_penalty = -0.1  # Reduced penalty for engine firing
        self.curving = 0.2

    def score_landing_path(self, x0, y0):
        # The flags on the map are between -0.2 to 0.2
        if x0 > 0.2 or x0 < -0.2:
            return -100

        inside_parabola = 0
        total_points = len(self.path) + 1

        for x, y in self.path:
            # y_parabola = self.a * x ** 2 + self.b * x + self.c
            # if y >= y_parabola:
            #     inside_parabola += 1
            vertical_distance = abs(y0 - y)
            max_horizontal_distance = np.sqrt(self.curving * vertical_distance)
            if abs(x - x0) <= max_horizontal_distance:
                inside_parabola += 1

        score = (inside_parabola / total_points) * 100
        return score

    # def descent_path_landing_score(self, x, y):
    #     # The flags on the map are between -0.2 to 0.2
    #     inside_parabola = 0
    #     a = 0.2
    #
    #     vertical_distance = abs(0 - y)
    #
    #     max_horizontal_distance = np.sqrt(a * vertical_distance)
    #
    #     if abs(x) <= max_horizontal_distance // 2:
    #         return 0
    #
    #     else:
    #         return - ((abs(x) - max_horizontal_distance // 2) ** 2)

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
        # observation, reward, done, _, info = self.env.step(action)
        # x, y = observation[0], observation[1]
        # self.safe_penalties.append(self.descent_path_landing_score(x, y))
        # self.path.append((x, y))
        # reward += self.safe_penalties[-1]
        # if done and self.debug:
        #     #print(self.safe_penalties)
        #     safePara.plot_landing_path(self.path, 0.167, x, y, sum(self.safe_penalties))
        # return observation, reward, done, _, info
        observation, reward, done, _, info = self.env.step(action)
        x, y = observation[0], observation[1]
        self.path.append((x, y))

        y_desired = self.a * x ** 2 + self.b * x + self.c
        altitude_scaling_factor = 1 - y / self.env.observation_space.high[1]  # Normalize using the max y from space

        # Reward shaping: only penalize if y is below y_desired
        if y < y_desired:
            deviation = y_desired - y
            penalty = -np.sqrt(
                deviation) * altitude_scaling_factor  # Calculate a scaled penalty that increases as the lander descends
            shaped_reward = max(self.max_penalty, penalty)  # Ensure the penalty does not exceed maximum allowed penalty
            reward += shaped_reward  # Update the original reward with the shaped reward
            if action in [1, 2, 3]:
                reward += self.reduced_engine_penalty + 0.3

        if done:
            info['safety'] = self.score_landing_path(x, y)

        if done and self.debug:
            SafePara.plot_landing_path(self.path, self.curving, x, y, info['safety'])

        return observation, reward, done, _, info
