from typing import List

from environment.Environement import Environment


class EpisodeRewardsWrapper:
    """
    Environement wrapper that tracks episode rewards.
    """

    def __init__(self, env: Environment) -> None:
        self.env = env
        self.episode_rewards = []
        self.current_episode_rewards = None
        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states

    def reset(self):
        # append last episode's rewards to list of all episode rewards
        if self.current_episode_rewards is not None:
            self.episode_rewards.append(self.current_episode_rewards)

        # reset current episode rewards
        self.current_episode_rewards = []

        return self.env.reset()

    def step(self, action):
        state, reward, done = self.env.step(action)

        # append reward to current episode rewards
        self.current_episode_rewards.append(reward)

        return state, reward, done

    def flush_rewards(self) -> List[List[float]]:
        all_rewards = self.episode_rewards + [self.current_episode_rewards]

        self.episode_rewards = []
        self.current_episode_rewards = None

        return all_rewards
