from typing import Tuple
import numpy as np
from environment.Environement import Environment


def e_greedy(Q: np.ndarray, s: int, epsilon: float, n_actions: int, random_state: np.random.RandomState) -> int:
    """
    Epsilon greedy policy.

    args:
        Q: Q-value table
        s: current state
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
    return np.argmax(Q[s, :])


def sarsa(env: Environment, max_episodes: int, eta: float, gamma: float, epsilon: float, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    SARSA Control algorithm.
    Updates Q table using tuples of (s, a, r, s_, a_)

    Q(s, a) <- Q(s, a) + a * [r + gamma * Q(s_, a_) - Q(s, a)]

    args:
        env: the environment
        max_episodes: maximum number of episodes
        eta: learning rate
        gamma: discount factor
        epsilon: exploration rate
        seed: random seed

    returns:
        policy, value-function
    """
    random_state = np.random.RandomState(seed)

    # initialise eta and epsilon decay
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    # initalise Q table
    Q = np.zeros((env.n_states, env.n_actions))

    # epsilon greedy shortcut
    def get_action(state: int, epsilon: float):
        return e_greedy(Q, state, epsilon, env.n_actions, random_state)

    for lr, e in zip(eta, epsilon):
        # get initial state and action
        s = env.reset()
        a = get_action(s, e)

        # run through episode
        done = False
        while not done:
            # get next state, reward and next action
            s_, r, done = env.step(a)
            a_ = get_action(s_, e)

            # update Q table
            Q[s, a] += lr * (r + gamma * Q[s_, a_] - Q[s, a])

            # move state and action pointers
            s = s_
            a = a_

    # compute policy and value-function from Q table
    policy = Q.argmax(axis=1)
    value = Q.max(axis=1)

    return policy, value


def q_learning(env: Environment, max_episodes: int, eta: float, gamma: float, epsilon: float, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Q Learning Control algorithm.
    Updates Q table using tuples of (s, a, r, s_)

    Q(s, a) <- Q(s, a) + a * [r + gamma * max_a_(Q(s_)) - Q(s, a)]

    args:
        env: the environment
        max_episodes: maximum number of episodes
        eta: learning rate
        gamma: discount factor
        epsilon: exploration rate
        seed: random seed

    returns:
        policy and value-function
    """

    random_state = np.random.RandomState(seed)

    # initialise eta and epsilon decay
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    # initalise Q table
    Q = np.zeros((env.n_states, env.n_actions))

    # epsilon greedy shortcut
    def get_action(state: int, epsilon: float):
        return e_greedy(
            Q, state, epsilon, env.n_actions, random_state)

    for lr, e in zip(eta, epsilon):
        s = env.reset()
        done = False

        # run through episode
        while not done:
            # get action, next state and reward
            a = get_action(s, e)
            s_, r, done = env.step(a)

            # update Q table
            # print(r, gamma, np.max(Q[s_]), Q[s, a])
            Q[s, a] += lr * (r + gamma * np.max(Q[s_]) - Q[s, a])

            # update state pointer
            s_ = s

    # compute policy and value-function from Q table
    policy = Q.argmax(axis=1)
    value = Q.max(axis=1)

    return policy, value
