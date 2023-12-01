import torch
import numpy as np

from environment.Environement import Environment
from agents.DeepQNetwork import DeepQNetwork
from agents.ReplayBuffer import ReplayBuffer


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
    """
    SARSA Control algorithm using linear approximation.
    Environemnt should be wrapped with LinearWrapper class

    Updates weights using tuples of (s, a, r, s_, a_)

    Q(s, a) = theta * Phi(s, a)

    theta <- theta + a * [r + gamma * max_a_(Q(s_, a_)) - Q(s, a)] * Phi(s, a)

    args:
        env: the environment
        max_episodes: maximum number of episodes
        eta: learning rate
        gamma: discount factor
        epsilon: exploration rate
        seed: random seed

    returns:
        weights of linear approximated Q table
    """
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
        # reset environment and compute intial q values and action
        features = env.reset()
        Q = features.dot(theta)
        a = get_action(Q, e)
        done = False

        # run through episode
        while not done:
            # get action, next state and reward
            features_, r, done = env.step(a)

            # update Q table
            Q_a = Q[a]
            Q = features_.dot(theta)

            # update weights using next action
            a_ = get_action(Q, e)
            temporal_dif = r + gamma * Q[a_] - Q_a
            theta += lr * temporal_dif * features[a]

            # move feature pointer
            features = features_

    return theta


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    """
    Q Learning Control algorithm using linear approximation.
    Environemnt should be wrapped with LinearWrapper class

    Updates weights using tuples of (s, a, r, s_)

    Q(s, a) = theta * Phi(s, a)

    theta <- theta + a * [r + gamma * max_a_(Q(s_,)) - Q(s, a)] * Phi(s, a)

    args:
        env: the environment
        max_episodes: maximum number of episodes
        eta: learning rate
        gamma: discount factor
        epsilon: exploration rate
        seed: random seed

    returns:
        weights of linear approximated Q table
    """
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

            # calculate new Q
            Q_a = Q[a]
            Q = features_.dot(theta)

            # update weights
            temporal_dif = r + gamma * np.max(Q) - Q_a
            theta += lr * temporal_dif * features[a]

            # move feature pointer
            features = features_

    return theta


def deep_q_network_learning(env: Environment,
                            max_episodes: int,
                            learning_rate: float,
                            gamma: float,
                            epsilon: float,
                            batch_size: int,
                            target_update_frequency: int,
                            buffer_size: int,
                            kernel_size: int,
                            conv_out_channels: int,
                            fc_out_features: int,
                            seed: int):
    """
    Deep Q Learning control algorithm.
    Trains a Deep Q Network which approximates the Q table

    args:
        env: the environment
        max_episodes: maximum number of episodes
        learning_rate: learning rate
        gamma: discount factor
        epsilon: exploration rate
        batch_size: batch size
        target_update_frequency: frequency of updating target network
        buffer_size: size of replay buffer
        kernel_size: size of convolutional kernel
        conv_out_channels: number of convolutional output channels
        fc_out_features: number of fully connected output features
        seed: random seed

    returns:
        the trained Deep Q Network
    """

    # initialise replay buffer, networks and epsiolon decay
    random_state = np.random.RandomState(seed)
    replay_buffer = ReplayBuffer(buffer_size, random_state)

    dqn = DeepQNetwork(env, learning_rate, kernel_size, conv_out_channels,
                       fc_out_features, seed=seed)
    tdqn = DeepQNetwork(env, learning_rate, kernel_size, conv_out_channels,
                        fc_out_features, seed=seed)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    # epsilon greedy using q network
    def e_greedy(state: np.ndarray, epsilon: float):
        # explore
        if random_state.rand() < epsilon:
            action = random_state.choice(env.n_actions)

        # exploit
        else:
            with torch.no_grad():
                q = dqn(np.array([state]))[0].numpy()

            qmax = max(q)
            best = [a for a in range(env.n_actions)
                    if np.allclose(qmax, q[a])]
            action = random_state.choice(best)

        return action

    for i, e in enumerate(epsilon):
        # reset environemnt
        s = env.reset()
        done = False

        # run through episode
        while not done:

            # get action, next state and reward
            a = e_greedy(s, e)
            s_, r, done = env.step(a)

            # add transition to replay buffer, move state pointer
            replay_buffer.append((s, a, r, s_, done))
            s = s_

            # train network
            if len(replay_buffer) >= batch_size:
                transitions = replay_buffer.draw(batch_size)
                dqn.train_step(transitions, gamma, tdqn)

        # update target network
        if (i % target_update_frequency) == 0:
            tdqn.load_state_dict(dqn.state_dict())

    return dqn
