import contextlib
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
        # returns = discounted sum of rewards
        rewards = np.array(episode)
        discount_factors = np.array(
            [gamma ** i for i in range(len(rewards))])
        discounted_return = np.sum(rewards * discount_factors)

        # add to list
        returns.append(discounted_return)

    return returns


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
