from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv


class PPOActor():
    def __init__(self, ckpt: str = None, environment: VecEnv = None, model=None):
        '''
          Requires environment to be a 1-vectorized environment

          The `ckpt` is a .zip file path that leads to the checkpoint you want
          to use for this particular actor.

          If the `model` variable is provided, then this constructor will store
          that as the internal representing model instead of loading one from the
          checkpoint path

        '''
        assert ckpt is not None or model is not None

        self.environment = environment
        if model is not None:
            self.model = model
            return

        self.model = PPO.load(ckpt, env=self.environment)

    def select_action(self, obs):
        '''
          Gives the action prediction of this particular actor
        '''
        action, _ = self.model.predict(observation=obs)
        return action
