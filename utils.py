import contextlib
from matplotlib import pyplot as plt
import numpy as np


from typing import List


@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def play(env):
    actions = ['w', 'a', 's', 'd']

    state = env.reset()
    env.render()

    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')

        state, r, done = env.step(actions.index(c))

        env.render()
        print('Reward: {0}.'.format(r))


def compute_episode_returns(episodes: List[List[float]], gamma: float) -> List[float]:
    """
    Computes the episode returns
    Returns = sum of discounted rewards

    args: 
        episodes: List of rewards obtained through each episode
        gamma: discount factor

    returns:
        list of returns for each episode
    """
    returns = []

    for episode in episodes:
        rewards = np.array(episode)

        # returns = discounted sum of rewards
        discount_factors = np.array(
            [gamma ** i for i in range(len(rewards))])
        discounted_return = np.sum(rewards * discount_factors)

        # add to list
        returns.append(discounted_return)

    return returns


def compute_rolling_average(values: List[float], window_length: int = 20) -> List[int]:
    """
    Computes the rolling average of a list of floats

    args:
        values: list of floats
        window_length: rolling average windown length

    returns:
        rolling average
    """
    return np.convolve(values, np.ones(
        window_length)/window_length, mode='valid')


def epsilon_greedy(Qs: np.ndarray, epsilon: float, n_actions: int, random_state: np.random.RandomState) -> int:
    """
    Epsilon greedy policy.

    args:
        Qs: Q-value table for the current state (numpy array with shape: (n_actions,))
        epsilon: exploration rate
        n_actions: number of actions
        random_state: random state

    returns:
        action
    """
    # explore
    if random_state.rand() < epsilon:
        return random_state.choice(n_actions)

    # exploit
    # break ties randomly by selecting random action that is close to max
    qmax = np.max(Qs)
    best = [a for a in range(n_actions) if np.allclose(qmax, Qs[a])]
    return random_state.choice(best)


def plot_episode_returns(returns: List[float], name: str, label: str = None, window_length: int = 20, ax=None, alpha=0.5) -> None:
    """
    Plots the returns (sum of discounted rewards) for each episode during training with a rolling window average

    args:
        returns: list of return for each episode
        title: the title of the plot
        label: the label of the line
        window_length: the rolling window length
        ax: optional axis to plot on
    """
    # get ax if none provided
    if ax is None:
        fig, ax = plt.subplot(1)
    # compute rolling average
    N = len(returns)
    episodes = np.arange(1, N+1)

    moving_average_episodes = episodes[window_length-1:]
    moving_average = compute_rolling_average(returns)

    # plot moving average returns
    ax.plot(moving_average_episodes,
            moving_average, label=label, alpha=alpha)

    # format plot
    ax.set_title(name)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Moving average of discounted Return')
    if label is not None:
        ax.legend()
