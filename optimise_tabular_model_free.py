import argparse
from environment.FrozenLake import BIG_LAKE, FrozenLake, SMALL_LAKE

from environment.FrozenLake import FrozenLake, SMALL_LAKE
from environment.EpisodeRewardsWrapper import EpisodeRewardsWrapper
from agents.tabular_model_rl import policy_iteration
from agents.tabular_model_free_rl import sarsa, q_learning
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys

from utils import compute_episode_returns, compute_rolling_average, plot_episode_returns


SEED = 0
GAMMA = 0.9
THETA = 0
MAX_EPISODES = 4000
THRESHOLD = 0.95
ENV_MAX_STEPS = 100

# paramters we will search through
LR = [1.4, 1.2, 1, 0.8, 0.6, 0.4, 0.2, 0.1]
EPSILON = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.01]

# lakes we are searching on
LAKE_NAMES = ["Big lake"]  # , "Big lake"]
NAME_TO_LAKE = {
    "Small lake": SMALL_LAKE,
    "Big lake": BIG_LAKE
}

# algorithms we are searching with
ALGORITHM_NAMES = ["SARSA", "Q-Learning"]
NAME_TO_ALGORITHM = {
    "SARSA": sarsa,
    "Q-Learning": q_learning
}


def build_env(lake, wrap=False):
    """
    Builds a frozen lake environment

    args:
        lake: the lake to build the environment from
        wrap: wrap the environment with an EpisodeRewardsWrapper

    returns:
        the environment
    """
    env = FrozenLake(lake, slip=0.1, max_steps=ENV_MAX_STEPS, seed=SEED)
    if not wrap:
        return env

    return EpisodeRewardsWrapper(env)


def get_optimal_return(lake):
    """
    Returns the returns (sum of discounted rewards) under an optimal policy
    Uses policy iteration to get this optimal policy

    args:
        lake: the lake to build the environment from

    returns:
        optimal return
    """
    env = build_env(lake)
    # policy iteration returns an optimal policy
    policy, _, _ = policy_iteration(
        env,
        GAMMA,
        theta=0,
        max_iterations=sys.maxsize)

    # run through episode with optimal policy
    wrapped_env = build_env(lake, wrap=True)
    s = wrapped_env.reset()
    done = False
    while not done:
        a = policy[s]
        s, _, done = wrapped_env.step(a)

    # return last episode's return
    rewards = wrapped_env.flush_rewards()
    returns = compute_episode_returns(rewards, GAMMA)

    return returns[-1]


def run_training(algorithm, lake, lr, epsilon):
    """
    Runs the training of an algorithm and calculated returns for each episode during training

    args:
        algorithm: SARSA or Q-learning function
        lake: the lake to build the environment with
        lr: the learning rate to use for training 
        epsilon: the value of epsilon to use for training

    returns:
        the returns for each episode during training
    """

    wrapped_env = build_env(lake, wrap=True)

    algorithm(wrapped_env,
              max_episodes=MAX_EPISODES,
              eta=lr,
              gamma=GAMMA,
              epsilon=epsilon,
              seed=SEED)

    rewards = wrapped_env.flush_rewards()

    return np.array(compute_episode_returns(rewards, GAMMA))


def search_parameter_space(lake, algorithm, optimal_return):
    """
    Performs a grid search of different learning rates and epsilon values

    args:
        lake: the lake to build the environment with
        algorithm: SARSA or Q-learning function
        optimal_return: the return obtained under an optimal policy

    returns:
        A tuple (results_grid, all_returns)
        where:
            results_grid: n x m array of the number of episodes required to reach the optimal returns
                        (n = number of learning rates, m = number of epsilon values)
            all_returns: n x m x max_episodes array of the returns obtained for each episode  
    """

    results = np.zeros((len(LR), len(EPSILON)))
    all_returns = np.zeros((len(LR), len(EPSILON), MAX_EPISODES))

    # grid search
    for i, lr in enumerate(LR):
        for j, epsilon in enumerate(EPSILON):

            # get returns for each episode in training
            episode_returns = run_training(
                algorithm, lake, lr, epsilon)
            rolling_returns = compute_rolling_average(episode_returns)

            # find index of first episode to acheive optimal return
            convergence_at = np.argmax(
                rolling_returns >= (optimal_return * THRESHOLD))

            print(
                f"lr: {lr}, epsilon: {epsilon} -> {rolling_returns.max()}({convergence_at})")

            # store result
            results[i, j] = convergence_at
            all_returns[i, j] = episode_returns

    # set 0 values (where it never reached to optimal) to be the maximum value (in order for plot to make sense)
    results[results == 0] = results.max()

    return results, all_returns


def plot_combination(results, ax, title):
    """
    Plots the result of a lake+algorithm combination on a heat map
    """
    sns.heatmap(results, linewidth=0.5, ax=ax)

    # set axis labels and ticks
    ax.set_xticks(np.arange(len(EPSILON)), EPSILON)
    ax.set_xlabel("Epsilon")
    ax.set_yticks(np.arange(len(LR)), LR)
    ax.set_ylabel('Learning Rate')

    ax.set_title(title)


def print_best_combination():
    # all combination of algorithms and lakes
    for lake_name in LAKE_NAMES:
        for algorithm_name in ALGORITHM_NAMES:
            # load results
            results_file_name, _ = get_filenames(
                lake_name, algorithm_name)
            results = np.load(results_file_name)

            # find best index
            best_lr_idx, best_epsilon_idx = np.unravel_index(
                results.argmin(), results.shape)

            # print best combination
            print(f"{lake_name} with {algorithm_name}")
            print(f"Best learning rate: {LR[best_lr_idx]}")
            print(f"Best epsilon: {EPSILON[best_epsilon_idx]}")
            print("\n\n")


def plot_best_combination_returns():
    fig, lake_axs = plt.subplots(1, len(LAKE_NAMES))

    if len(LAKE_NAMES) == 1:
        lake_axs = np.array([lake_axs])

    for lake_name, ax in zip(LAKE_NAMES, lake_axs):
        # plot optimal return as horizontal line
        optimal_return = get_optimal_return(NAME_TO_LAKE[lake_name])
        ax.plot(np.arange(MAX_EPISODES), [optimal_return for _ in range(MAX_EPISODES)],
                color='red', label="Optimal")

        for algorithm_name in ALGORITHM_NAMES:
            # load results and returns
            results_file_name, returns_filename = get_filenames(
                lake_name, algorithm_name)
            returns = np.load(returns_filename)
            results = np.load(results_file_name)

            # find best returns and plot
            best_lr_idx, best_epsilon_idx = np.unravel_index(
                results.argmin(), results.shape)
            best_returns = returns[best_lr_idx, best_epsilon_idx]

            plot_episode_returns(best_returns, lake_name,
                                 f"{algorithm_name} lr:{LR[best_lr_idx]}, eps:{EPSILON[best_epsilon_idx]}", ax=ax)

            # find worse and plot
            min_returns = returns.min(axis=2)
            worse_lr_idx, worse_epsilon_idx = np.unravel_index(
                min_returns.argmin(), results.shape)
            worse_returns = returns[worse_lr_idx, worse_epsilon_idx]

            plot_episode_returns(worse_returns, f"{lake_name} best and worse parameters",
                                 f"{algorithm_name} lr:{LR[worse_lr_idx]}, eps:{EPSILON[worse_epsilon_idx]}", ax=ax)

    fig.tight_layout()
    plt.show()


def display_results_heatmap():
    fig, ax = plt.subplots(len(LAKE_NAMES), len(ALGORITHM_NAMES))
    if len(LAKE_NAMES) == 1:
        ax = ax[np.newaxis, :]

    # all combination of algorithms and lakes
    for li, lake_name in enumerate(LAKE_NAMES):
        for ai, algorithm_name in enumerate(ALGORITHM_NAMES):
            # load results
            results_file_name, _ = get_filenames(
                lake_name, algorithm_name)
            results = np.load(results_file_name)

            # plot
            plot_combination(results, ax[li, ai],
                             f"{algorithm_name} on {lake_name}")

    fig.tight_layout()
    plt.show()


def get_filenames(lake_name, algorithm_name):
    results_file_name = f"search_results/{lake_name}_{algorithm_name}_results.npy"
    returns_file_name = f"search_results/{lake_name}_{algorithm_name}_returns.npy"

    return results_file_name, returns_file_name


def search_algorithm_on_lake(algorithm, algorithm_name, lake, lake_name):
    """
    Runs the parameter search for an algorithm-lake combination
    Will save the results to file

    args:
        algorithm: the function to run (SARSA or Q-Learning)
        algorithm_name: the name of the algorithm
        lake: the lake to build the environment with
        lake_name: the name of the lake
    """
    # compute optimal return
    optimal_return = get_optimal_return(lake)

    print(f"{lake_name} - {algorithm_name} (optimal return :{optimal_return})")

    # run grid search for lake and algorithm combination
    results, all_returns = search_parameter_space(
        lake, algorithm, optimal_return)

    # save to file
    results_file_name, returns_file_name = get_filenames(
        lake_name, algorithm_name)
    np.save(results_file_name, results)
    np.save(returns_file_name, all_returns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', type=str, default=None,
                        choices=ALGORITHM_NAMES, help="The Algorithm to run")
    parser.add_argument('-l', type=str, default=None,
                        choices=LAKE_NAMES, help="The Lake to run on")
    args = parser.parse_args()

    # if algorithm or lake not provided, display results
    if args.a is None or args.l is None:
        print("Algorithm or Lake not provided, will present results instead of running search...")
        display_results_heatmap()
        plot_best_combination_returns()
        print_best_combination()
        exit()

    # else run search
    search_algorithm_on_lake(
        NAME_TO_ALGORITHM[args.a], args.a, NAME_TO_LAKE[args.l], args.l)
