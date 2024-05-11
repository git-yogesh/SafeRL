import torch
import random
import numpy as np
import gymnasium as gym
import cv2


from tqdm import tqdm, trange


def reseed(seed, env=None):
    '''
        Sets the seed for reproducibility

        When @param env is provided, also sets the
        random number generataor of the gym environment
        to this particular seed
    '''
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if env is not None:
        env.unwrapped._np_random = gym.utils.seeding.np_random(seed)[0]


# Visualize
def visualize(env, algorithm=None, video_name="test"):
    """Visualize a policy network for a given algorithm on a single episode

        Args:
            env_name: Name of the gym environment to roll out `algorithm` in, it will be instantiated using gym.make or make_vec_env
            algorithm (PPOActor): Actor whose policy network will be rolled out for the episode. If
            no algorithm is passed in, a random policy will be visualized.
            video_name (str): Name for the mp4 file of the episode that will be saved (omit .mp4). Only used
            when running on local machine.
    """

    def get_action(obs):
        if not algorithm:
            return env.action_space.sample()
        else:
            return algorithm.select_action(obs)

    video = cv2.VideoWriter(f"{video_name}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 24, (600, 400))

    obs = env.reset()

    total = 0
    for i in range(500):
        action = get_action(obs)
        res = env.step(action)
        obs, reward, done, info = res

        if done:
            break

        im = env.render()
        im = im[:, :, ::-1]

        video.write(im)
        total += reward

    video.release()
    env.close()
    print(f"Video saved as {video_name}.mp4. Reward: {total}")


# Evaluate Policy
def evaluate_policy(actor, environment, num_episodes=100, progress=True):
    '''
        Returns the mean trajectory reward of rolling out actor on `environment

        Parameters
        - actor: PPOActor instance, defined in Part 1
        - environment: classstable_baselines3.common.vec_env.VecEnv instance
        - num_episodes: total number of trajectories to collect and average over
    '''
    total_rew = 0
    rewards = []
    episode_rew = 0
    iterate = (trange(num_episodes) if progress else range(num_episodes))
    for _ in iterate:
        obs = environment.reset()
        done = False
        episode_rew = 0
        while not done:
            action = actor.select_action(obs)

            next_obs, reward, done, info = environment.step(action)
            total_rew += reward
            episode_rew += reward
            obs = next_obs
            # done = done.any() if isinstance(done, np.ndarray) else done

        rewards.append(episode_rew)
    return (total_rew / num_episodes).item(), rewards