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
