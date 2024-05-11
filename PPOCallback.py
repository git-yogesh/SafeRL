from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from PPOActor import PPOActor
from utils import evaluate_policy


class PPOCallback(BaseCallback):
    def __init__(self, verbose=0, save_path='default', eval_env=None):
        super(PPOCallback, self).__init__(verbose)
        self.rewards = []

        self.save_freq = 120000
        self.min_reward = -np.inf
        self.actor = None
        self.eval_env = eval_env

        self.save_path = save_path

        self.eval_steps = []
        self.eval_rewards = []

    def _init_callback(self) -> None:
        pass

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """

        self.actor = PPOActor(model=self.model)

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """

        episode_info = self.model.ep_info_buffer
        rewards = [ep_info['r'] for ep_info in episode_info]
        mean_rewards = np.mean(rewards)

        self.rewards.append(mean_rewards)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        if self.eval_env is None:
            return True

        if self.num_timesteps % self.save_freq == 0 and self.num_timesteps != 0:
            mean_reward, _ = evaluate_policy(self.actor, environment=self.eval_env, num_episodes=20)
            print(f'evaluating {self.num_timesteps=}, {mean_reward=}=======')

            self.eval_steps.append(self.num_timesteps)
            self.eval_rewards.append(mean_reward)
            if mean_reward > self.min_reward:
                self.min_reward = mean_reward
                self.model.save(self.save_path)
                print(f'model saved on eval reward: {self.min_reward}')

        return True

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        print(f'model saved on eval reward: {self.min_reward}')

        plt.plot(self.eval_steps, self.eval_rewards, c='red')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('Rewards over Episodes')

        plt.show()
        plt.close()
