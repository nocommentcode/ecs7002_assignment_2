from environment.FrozenLake import FrozenLake, SMALL_LAKE
from environment.FrozenLakeImageWrapper import FrozenLakeImageWrapper
from environment.LinearWrapper import LinearWrapper
from environment.EpisodeRewardsWrapper import EpisodeRewardsWrapper
from agents.tabular_model_free_rl import sarsa, q_learning
from agents.non_tabular_model_free_rl import deep_q_network_learning, linear_sarsa, linear_q_learning
import matplotlib.pyplot as plt
from utils import compute_episode_returns, plot_episode_returns

import os
# for some reason this is needed for it not to crash?
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    seed = 0
    lake = SMALL_LAKE
    gamma = 0.9
    max_episodes = 4000
    eta = 0.5
    epsilon = 0.5
    learning_rate = 0.001
    epsilon = 0.2
    batch_size = 32
    target_update_frequency = 4
    buffer_size = 256
    kernel_size = 3
    conv_out_channels = 4
    fc_out_features = 8

    args = (max_episodes, eta, gamma, epsilon, seed)
    dqn_args = (max_episodes, learning_rate, gamma, epsilon, batch_size,
                target_update_frequency, buffer_size,
                kernel_size, conv_out_channels,
                fc_out_features, seed)

    names = ("SARSA", "Q-Learning", "Linear SARSA",
             "Linear Q-Learning", "Deep Q-Network")

    algorithms = [sarsa, q_learning, linear_sarsa,
                  linear_q_learning, deep_q_network_learning]

    arguments = (args, args, args, args, dqn_args)

    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    wrapped_envs = (EpisodeRewardsWrapper(env),
                    EpisodeRewardsWrapper(env),
                    EpisodeRewardsWrapper(LinearWrapper(env)),
                    EpisodeRewardsWrapper(LinearWrapper(env)),
                    EpisodeRewardsWrapper(FrozenLakeImageWrapper(env)))

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 4)
    ax4 = fig.add_subplot(2, 3, 5)
    ax5 = fig.add_subplot(1, 3, 3)

    flatt_ax = [ax1, ax2, ax3, ax4, ax5]

    for name, algorithm, arg, ax, wrapped_env in zip(names, algorithms, arguments, flatt_ax, wrapped_envs):
        algorithm(wrapped_env, *arg)
        rewards = wrapped_env.flush_rewards()
        returns = compute_episode_returns(rewards, gamma)
        plot_episode_returns(returns, name, ax=ax, alpha=1)

    plt.suptitle('Moving average of discounted Rewards for different agents')
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
