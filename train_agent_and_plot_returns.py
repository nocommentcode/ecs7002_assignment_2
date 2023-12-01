
from environment.FrozenLake import FrozenLake, SMALL_LAKE
from environment.FrozenLakeImageWrapper import FrozenLakeImageWrapper
from environment.LinearWrapper import LinearWrapper
from environment.EpisodeRewardsWrapper import EpisodeRewardsWrapper
from agents.tabular_model_rl import policy_iteration, value_iteration
from agents.tabular_model_free_rl import sarsa, q_learning
from agents.non_tabular_model_free_rl import deep_q_network_learning, linear_sarsa, linear_q_learning
import matplotlib.pyplot as plt
from typing import List
import numpy as np

SEED = 1


def compute_episode_returns(episodes: List[List[float]], gamma: float) -> List[float]:
    returns = []

    for episode in episodes:
        # returns = discounted sum of rewards
        rewards = np.array(episode)
        discount_factors = np.array(
            [gamma ** i for i in range(len(rewards))])
        discounted_return = np.sum(rewards * discount_factors)

        # add to list
        returns.append(discounted_return)

    return returns


def plot_episode_returns(returns: List[float], name: str, window_length: int = 20, ax=None) -> None:
    if ax is None:
        fig, ax = plt.subplot(1)

    N = len(returns)
    episodes = np.arange(1, N+1)

    ax.plot(episodes, returns, label='Episode Returns', alpha=0.3)

    moving_average_episodes = episodes[window_length-1:]
    moving_average = np.convolve(returns, np.ones(
        window_length)/window_length, mode='valid')

    ax.plot(moving_average_episodes,
            moving_average, label='Moving Average')

    ax.set_title(name)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Discounted Return')
    ax.legend()


def run_sarsa(env, ax):
    wrapped_env = EpisodeRewardsWrapper(env)
    gamma = 0.9
    epsilon = 0.5
    eta = 0.5
    max_episodes = 4000

    sarsa(wrapped_env, max_episodes, eta=eta,
          gamma=gamma, epsilon=epsilon, seed=SEED)
    rewards = wrapped_env.flush_rewards()
    returns = compute_episode_returns(rewards, gamma)
    plot_episode_returns(returns, "SARSA", ax=ax)


def run_q_learning(env, ax):
    wrapped_env = EpisodeRewardsWrapper(env)
    gamma = 0.9
    epsilon = 0.5
    eta = 0.5
    max_episodes = 4000

    q_learning(wrapped_env, max_episodes, eta=eta,
               gamma=gamma, epsilon=epsilon, seed=SEED)
    rewards = wrapped_env.flush_rewards()
    returns = compute_episode_returns(rewards, gamma)
    plot_episode_returns(returns, "Q-Learning", ax=ax)


def run_linear_sarsa(env, ax):
    linear_env = LinearWrapper(env)
    wrapped_env = EpisodeRewardsWrapper(linear_env)
    wrapped_env.n_features = linear_env.n_features
    gamma = 0.9
    epsilon = 0.5
    eta = 0.5
    max_episodes = 4000

    linear_sarsa(wrapped_env, max_episodes, eta=eta, gamma=gamma,
                 epsilon=epsilon, seed=SEED)
    rewards = wrapped_env.flush_rewards()
    returns = compute_episode_returns(rewards, gamma)
    plot_episode_returns(returns, "Linear SARSA", ax=ax)


def run_linear_q_learning(env, ax):
    linear_env = LinearWrapper(env)
    wrapped_env = EpisodeRewardsWrapper(linear_env)
    wrapped_env.n_features = linear_env.n_features

    gamma = 0.9
    epsilon = 0.5
    eta = 0.5
    max_episodes = 4000

    linear_q_learning(wrapped_env, max_episodes, eta=eta, gamma=gamma,
                      epsilon=epsilon, seed=SEED)
    rewards = wrapped_env.flush_rewards()
    returns = compute_episode_returns(rewards, gamma)
    plot_episode_returns(returns, "Linear Q-Learning", ax=ax)


def run_dqn(env, ax):
    image_env = FrozenLakeImageWrapper(env)
    wrapped_env = EpisodeRewardsWrapper(image_env)
    wrapped_env.state_shape = image_env.state_shape

    gamma = 0.9
    max_episodes = 4000

    deep_q_network_learning(wrapped_env, max_episodes, learning_rate=0.001,
                            gamma=gamma,  epsilon=0.2, batch_size=32,
                            target_update_frequency=4, buffer_size=256,
                            kernel_size=3, conv_out_channels=4,
                            fc_out_features=8, seed=4)
    rewards = wrapped_env.flush_rewards()
    returns = compute_episode_returns(rewards, gamma)
    plot_episode_returns(returns, "DQN", ax=ax)


def main():
    lake = SMALL_LAKE
    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=SEED)

    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    # sarsa, q-learning, linear sarsa, linear q-learning, dqn
    algorithms = [run_sarsa, run_q_learning,
                  run_linear_sarsa, run_linear_q_learning, run_dqn]
    flatt_ax = ax.flatten()

    for algorithm, ax in zip(algorithms, flatt_ax):
        algorithm(env, ax)

    plt.suptitle('Discounted Rewards for different agents')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
