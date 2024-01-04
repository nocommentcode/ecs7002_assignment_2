from environment.FrozenLake import BIG_LAKE, FrozenLake, SMALL_LAKE

from environment.FrozenLake import FrozenLake, SMALL_LAKE
from environment.EpisodeRewardsWrapper import EpisodeRewardsWrapper
from agents.tabular_model_rl import policy_iteration
from agents.tabular_model_free_rl import sarsa, q_learning
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from train_agent_and_plot_returns import compute_episode_returns


SEED = 0
GAMMA = 0.9
MAX_ITER = 10**100
THETA = 0
MAX_EPISODES = 10000
LR = [1, 0.9, 0.8, 0.7, 0.6]
EPSILON = [1, 0.9, 0.8, 0.7, 0.6]
REPETITIONS = 1


def get_optimal_return(env):
    # policy iteration returns an optimal policy
    policy, _, _ = policy_iteration(env, GAMMA, 0, MAX_ITER)

    # run through episode with optimal policy
    wrapped_env = EpisodeRewardsWrapper(env)

    s = wrapped_env.reset()
    done = False
    while not done:
        a = policy[s]
        s, _, done = wrapped_env.step(a)

    rewards = wrapped_env.flush_rewards()
    return compute_episode_returns(rewards, GAMMA)[-1]


def run_training(algorithm, env, lr, epsilon):
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
    returns = []
    for _ in range(REPETITIONS):
        returns.append(run_training(algorithm, env, lr, epsilon))
    returns = np.array(returns)

    return returns.mean(0)


def search_parameter_space(env, algorithm, optimal_return):
    results = np.zeros((len(LR), len(EPSILON)))

    for i, lr in enumerate(LR):
        for j, epsilon in enumerate(EPSILON):
            episode_returns = run_multiple_trainings(
                algorithm, env, lr, epsilon)
            convergence_at = np.argmax(
                episode_returns >= (optimal_return - 0.01))
            print(
                f"lr: {lr}, epsilon: {epsilon} -> {episode_returns.max()}({convergence_at})")
            results[i, j] = convergence_at

    results[results == 0] = results.max()
    return results


def plot_results(results, ax, title):
    sns.heatmap(results, linewidth=0.5, ax=ax)

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
    for l, (lake_name, lake) in enumerate(zip(lake_names, lakes)):
        env = FrozenLake(lake, slip=0.1, max_steps=200, seed=SEED)
        optimal_return = get_optimal_return(env)
        print(f"Searching {lake_name} (optimal return :{optimal_return})")

        for a, (algo_name, algo) in enumerate(zip(algorithm_names, algorithms)):
            print(algo_name)
            results = search_parameter_space(env, algo, optimal_return)
            best_lr, best_epsilon = np.unravel_index(
                results.argmin(), results.shape)
            print(f"Best learning rate: {LR[best_lr]}")
            print(f"Best epsilon: {EPSILON[best_epsilon]}")
            plot_results(results, ax[l, a], f"{algo_name} on {lake_name}")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
