
from environment.FrozenLake import FrozenLake, SMALL_LAKE
from environment.FrozenLakeImageWrapper import FrozenLakeImageWrapper
from environment.LinearWrapper import LinearWrapper
from environment.EpisodeRewardsWrapper import EpisodeRewardsWrapper
from agents.tabular_model_free_rl import sarsa, q_learning
from agents.non_tabular_model_free_rl import deep_q_network_learning, linear_sarsa, linear_q_learning
import matplotlib.pyplot as plt
import numpy as np

from utils import compute_episode_returns, plot_episode_returns

SEED = 1


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
