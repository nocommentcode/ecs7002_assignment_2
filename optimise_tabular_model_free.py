from environment.FrozenLake import BIG_LAKE, FrozenLake, SMALL_LAKE

from environment.FrozenLake import FrozenLake, SMALL_LAKE
from environment.EpisodeRewardsWrapper import EpisodeRewardsWrapper
from agents.tabular_model_rl import policy_iteration
from agents.tabular_model_free_rl import sarsa, q_learning
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

from utils import compute_episode_returns


SEED = 0
GAMMA = 0.9
THETA = 0
MAX_EPISODES = 10000

# number of repetitions to average across
REPETITIONS = 1

# paramters we will search through
LR = [1, 0.9, 0.8, 0.7, 0.6]
EPSILON = [1, 0.9, 0.8, 0.7, 0.6]


def get_optimal_return(env):
    """
    Returns the returns (sum of discounted rewards) under an optimal policy
    """
    # policy iteration returns an optimal policy
    policy, _, _ = policy_iteration(
        env,
        GAMMA,
        theta=0,
        max_iterations=sys.maxsize)

    # run through episode with optimal policy
    wrapped_env = EpisodeRewardsWrapper(env)
    s = wrapped_env.reset()
    done = False
    while not done:
        a = policy[s]
        s, _, done = wrapped_env.step(a)

    # return last episode's return
    rewards = wrapped_env.flush_rewards()
    return compute_episode_returns(rewards, GAMMA)[-1]


def run_training(algorithm, env, lr, epsilon):
    """
    Runs the training of an algorithm and calculated returns for each episode during training

    args:
        algorithm: SARSA or Q-learning function
        env: frozen lake environment
        lr: the learning rate to use for training 
        epsilon: the value of epsilon to use for training

    returns:
        the returns for each episode during training
    """

    wrapped_env = EpisodeRewardsWrapper(env)

    algorithm(wrapped_env,
              max_episodes=MAX_EPISODES,
              eta=lr,
              gamma=GAMMA,
              epsilon=epsilon,
              seed=SEED)

    rewards = wrapped_env.flush_rewards()

    return compute_episode_returns(rewards, GAMMA)


def run_multiple_trainings(algorithm, env, lr, epsilon):
    """
    Runs the training of an algorithm multiple times and averages the results

    args:
        algorithm: SARSA or Q-learning function
        env: frozen lake environment
        lr: the learning rate to use for training 
        epsilon: the value of epsilon to use for training

    returns:
        the returns for each episode during training
    """
    returns = []
    for _ in range(REPETITIONS):
        returns.append(run_training(algorithm, env, lr, epsilon))
    returns = np.array(returns)

    return returns.mean(0)


def search_parameter_space(env, algorithm, optimal_return):
    """
    Performs a grid search of different learning rates and epsilon values

    args:
        env: frozen lake environment
        algorithm: SARSA or Q-learning function
        optimal_return: the return obtained under an optimal policy

    returns:
        results of the grid search a n x m array of the number of episodes required to reach the optimal returns
        (n = number of learning rates, m = number of epsilon values)
    """
    results = np.zeros((len(LR), len(EPSILON)))

    # grid search
    for i, lr in enumerate(LR):
        for j, epsilon in enumerate(EPSILON):

            # get returns for each episode in training
            episode_returns = run_multiple_trainings(
                algorithm, env, lr, epsilon)

            # find index of first episode to acheive optimal return
            convergence_at = np.argmax(
                episode_returns >= (optimal_return - 0.01))

            print(
                f"lr: {lr}, epsilon: {epsilon} -> {episode_returns.max()}({convergence_at})")

            # store result
            results[i, j] = convergence_at

    # set 0 values (where it never reached to optimal) to be the maximum value (in order for plot to make sense)
    results[results == 0] = results.max()

    return results


def plot_results(results, ax, title):
    """
    Plots the results on a heat map
    """
    sns.heatmap(results, linewidth=0.5, ax=ax)

    # set axis labels and ticks
    ax.set_xticks(np.arange(len(EPSILON)), EPSILON)
    ax.set_xlabel("Epsilon")
    ax.set_yticks(np.arange(len(LR)), LR)
    ax.set_ylabel('Learning Rate')

    ax.set_title(title)


def main():
    lakes = [SMALL_LAKE, BIG_LAKE]
    lake_names = ["Small lake", "Big lake"]

    algorithms = [sarsa, q_learning]
    algorithm_names = ["SARSA", "Q-Learning"]

    fig, ax = plt.subplots(2, 2)

    # all combination of algorithms and lakes
    for l, (lake_name, lake) in enumerate(zip(lake_names, lakes)):

        # build lake and compute optimal return
        env = FrozenLake(lake, slip=0.1, max_steps=200, seed=SEED)
        optimal_return = get_optimal_return(env)
        print(f"Searching {lake_name} (optimal return :{optimal_return})")

        for a, (algo_name, algo) in enumerate(zip(algorithm_names, algorithms)):
            print(algo_name)

            # run grid search for lake and algorithm combination
            results = search_parameter_space(env, algo, optimal_return)

            # print best combination of lr and epsilon
            best_lr_idx, best_epsilon_idx = np.unravel_index(
                results.argmin(), results.shape)
            print(f"Best learning rate: {LR[best_lr_idx]}")
            print(f"Best epsilon: {EPSILON[best_epsilon_idx]}")
            print("\n\n")

            # plot heatmap
            plot_results(results, ax[l, a], f"{algo_name} on {lake_name}")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
