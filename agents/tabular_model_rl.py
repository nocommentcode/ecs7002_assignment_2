from typing import Tuple
import numpy as np

from environment.Environement import Environment


def policy_evaluation(env: Environment, policy: np.ndarray, gamma: float, theta: float, max_iterations: int) -> np.ndarray:
    """
    In-place iterative policy evaluation that will compute the state-value function for a given deterministic policy

    Vs <- sum(p(s_)*[r + gamma * Vs_]) for all s_

    args:
        env: the environment
        policy: the policy to evaluate
        gamma: the discount factor
        theta: the convergence thresholds
        max_iterations: the maximum number of iterations

    returns:
        the state-value function for the given policy
    """

    # initialise value to 0 for all states
    V = np.zeros(env.n_states, dtype=float)

    # run untill converged or max iterations reached
    for _ in range(max_iterations):
        delta = 0

        for s in range(env.n_states):
            v = V[s]
            # action under policy
            a = policy[s]

            # compute probabilities and rewards
            S_ = [s_ for s_ in range(env.n_states)]
            P = [env.p(s_, s, a) for s_ in S_]
            R = [env.r(s_, s, a) for s_ in S_]

            # update value
            V[s] = sum([p * (r + gamma * V[s_]) for s_, p, r in zip(S_, P, R)])

            delta = max(delta, abs(v - V[s]))

        # check for convergance
        if delta <= theta:
            break

    return V


def policy_improvement(env: Environment, V: np.ndarray, gamma: float) -> np.ndarray:
    """
    Policy improvent via Policy Improvement Theorem:

    P_ = argmax_a(sum( P(s_)*[r + gamma * V_]) for all s_)

    args:
        env: the environment
        V: the state-value function
        gamma: the discount factor

    returns:
        the improved policy
    """
    # the current policy
    policy = np.zeros(env.n_states, dtype=int)

    for s in range(env.n_states):
        action_values = np.zeros(env.n_actions, dtype=np.float32)
        for a in range(env.n_actions):
            # compute probabilities and rewards
            S_ = [s_ for s_ in range(env.n_states)]
            P = [env.p(s_, s, a) for s_ in S_]
            R = [env.r(s_, s, a) for s_ in S_]

            # calculate value
            action_values[a] = sum([p * (r + gamma * V[s_])
                                    for s_, p, r in zip(S_, P, R)])

        # policy gets argmax of action values
        policy[s] = np.argmax(action_values)

    return policy


def policy_iteration(env: Environment, gamma: float, theta: float, max_iterations: int, policy: np.ndarray = None, iteration: int = 0) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    In-place iterative deterministic policy evaluation.
    Interleaves policy evaluation and policy improvement.
    Will run recursively untill policy converges and return optimal policy and state-value function.

    args:
        env: the environment
        gamma: the discount factor
        theta: the convergence threshold
        max_iterations: the maximum number of iterations
        policy: the policy to evaluate

    returns:
        the optimal policy
        the state-value function
        the number of iterations untill covergence
    """
    # initialise policy if not supplied
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    # policy evaluation
    V = policy_evaluation(env, policy, gamma, theta, max_iterations)

    # policy improvement
    improved_policy = policy_improvement(env, V, gamma)

    # check for convergance
    if np.all(policy == improved_policy):
        return policy, V, iteration

    # reached maximum iterations
    if iteration >= max_iterations:
        return policy, V, iteration

    # recursive call
    return policy_iteration(env, gamma, theta, max_iterations, improved_policy, iteration+1)


def value_iteration(env: Environment, gamma: float, theta: float, max_iterations: int, V: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    In-place iterative value iteration.
    Will run untill convergance or max iterations and return optimal policy and state-value function.

    Vs <- max_a(sum(p(s_)*[r + gamma * Vs_]) for all s_)

    args:
        env: the environment
        gamma: the discount factor
        theta: the convergence threshold
        max_iterations: the maximum number of iterations
        V: the state-value function

    returns:
        the optimal policy and the state-value function
    """
    # initialise value if not provided
    if V is None:
        V = np.zeros(env.n_states, dtype=np.float32)
    else:
        V = np.array(V, dtype=float)

    # run untill converged or max iterations reached
    for iter in range(max_iterations):
        delta = 0

        for s in range(env.n_states):
            v = V[s]
            action_values = np.zeros(env.n_actions, dtype=np.float32)
            for a in range(env.n_actions):
                # compute probabilities and rewards
                S_ = [s_ for s_ in range(env.n_states)]
                P = [env.p(s_, s, a) for s_ in S_]
                R = [env.r(s_, s, a) for s_ in S_]

                # compute value
                action_values[a] = sum([p * (r + gamma * V[s_])
                                        for s_, p, r in zip(S_, P, R)])

            V[s] = action_values.max()
            delta = max(delta, abs(v - V[s]))

        # check for convergance
        if delta <= theta:
            break

    # build policy
    policy = policy_improvement(env, V, gamma)

    return policy, V, iter + 1
