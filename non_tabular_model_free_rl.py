from Environement import Environment
import numpy as np


def e_greedy(Q: np.ndarray, epsilon: float, n_actions: int, random_state: np.random.RandomState) -> int:
    """
    Epsilon greedy policy for linear approximated Q table

    args:
        Q: the approximate Q-value table for the current state
        epsilon: exploration rate
        n_actions: number of actions
        random_state: random state

    returns:
        action
    """
    # explore
    if random_state.rand() < epsilon:
        return random_state.choice(n_actions)

    # exploit (break ties randomly)
    qmax = np.max(Q)
    best = [a for a in range(n_actions) if np.allclose(qmax, Q[a])]
    return random_state.choice(best)


def linear_sarsa(env: Environment, max_episodes: int, eta: float, gamma: float, epsilon: float, seed: int = None) -> np.ndarray:
    random_state = np.random.RandomState(seed)

    # initialise eta and epsilon decay
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    # initialise weights
    theta = np.zeros(env.n_features)

    # epsilon greedy shortcut
    def get_action(Q: np.ndarray, epsilon: float):
        return e_greedy(Q, epsilon, env.n_actions, random_state)

    for lr, e in zip(eta, epsilon):
        # reset environment and compute intial q values
        features = env.reset()
        Q = features.dot(theta)
        a = get_action(Q, e)
        done = False

        # run through episode
        while not done:
            # get action, next state and reward
            features_, r, done = env.step(a)

            # calculate delta
            delta = r - Q[a]

            # update Q table
            Q = features_.dot(theta)

            # get next action
            a_ = get_action(Q, e)
            delta += gamma * Q[a_]

            # update weights
            theta += lr * delta * features[a]

            # move feature pointer
            features = features_

    return theta


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    # initialise eta and epsilon decay
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    # initialise weights
    theta = np.zeros(env.n_features)

    # epsilon greedy shortcut
    def get_action(Q: np.ndarray, epsilon: float):
        return e_greedy(Q, epsilon, env.n_actions, random_state)

    for lr, e in zip(eta, epsilon):
        # reset environment and compute intial q values
        features = env.reset()
        Q = features.dot(theta)
        done = False

        # run through episode
        while not done:
            # get action, next state and reward
            a = get_action(Q, e)
            features_, r, done = env.step(a)

            # calculate delta
            delta = r - Q[a]
            Q = features_.dot(theta)
            delta += gamma * np.max(Q)

            # update weights
            theta += lr * delta * features[a]

            # move feature pointer
            features = features_

    return theta
