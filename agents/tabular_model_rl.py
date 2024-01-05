from typing import Tuple
import numpy as np

from environment.Environement import Environment


def policy_evaluation(env: Environment, policy: np.ndarray, gamma: float, theta: float, max_iterations: int) -> (np.ndarray, int):
    """
    In-place iterative policy evaluation that will compute the state-value function for a given policy

    Note that the policy is concidered to be deterministic, (maps each state to a certain action)
    Hence, `policy` is a 1D array with each element mapping from a state to an action.

    Update step:
        Vs <- sum_s_[ P(s_ |s, a) * ( R(s_|s, a) + gamma * Vs_) ]
        where:
            a - the action from state s under the policy
            P(s_ |s, a) - probability of transitioning to state s_ after taking action a from state s
            R(s_|s, a) - the reward after taking a and transitioning to s_ 
            gamma - discount factor
            Vs_ - the next state's value

    args:
        env: the environment
        policy: the policy to evaluate (numpy array with shape: (n_states,))
        gamma: the discount factor
        theta: the convergence threshold
        max_iterations: the maximum number of iterations

    returns:
        A tuple (V, iter)
        where:
            V - the state-value function for the given policy (numpy array with shape: (n_states,))
            iter - the number of iterations it ran for
    """

    # initialise value to 0 for all states
    V = np.zeros(env.n_states, dtype=float)

    # Policy evaulation loop
    for iter in range(max_iterations):
        delta = 0

        # do 1 round of update for each state
        for s in range(env.n_states):
            v = V[s]

            # action under policy
            a = policy[s]

            # get next states, compute probabilities and rewards
            S_ = [s_ for s_ in range(env.n_states)]
            P = [env.p(s_, s, a) for s_ in S_]
            R = [env.r(s_, s, a) for s_ in S_]

            # Vs <- sum_s_[ P(s_ |s, a) * ( R(s_|s, a) + gamma * Vs_) ]
            V[s] = sum([p * (r + gamma * V[s_]) for s_, p, r in zip(S_, P, R)])

            delta = max(delta, abs(v - V[s]))

        # check for convergance
        if delta <= theta:
            break

    return V, iter


def policy_improvement(env: Environment, V: np.ndarray, gamma: float) -> np.ndarray:
    """
    Policy improvent via Policy Improvement Theorem.
    Improve on the previously used policy by building a policy that maximise the reward in each state.
    This function can also be used to build a policy from a value function

    Note that the policy is concidered to be deterministic, (maps each state to a certain action)
    Hence, `policy` is a 1D array with each element mapping from a state to an action.

    New policy at state s is build by:

    policy[s] <- argmax_a{ sum_s_[ P(s_ |s, a) * ( R(s_|s, a) + gamma * Vs_) ] }
    where:
        P(s_ |s, a) - probability of transitioning to state s_ after taking action a from state s
        R(s_|s, a) - the reward after taking a and transitioning to s_ 
        gamma - discount factor
        Vs_ - the next state's value

    args:
        env: the environment
        V: the state-value function (numpy array with shape: (n_states,))
        gamma: the discount factor

    returns:
        the improved policy (numpy array with shape: (n_states,))
    """
    # the current policy
    policy = np.zeros(env.n_states, dtype=int)

    # improve policy for each state
    for s in range(env.n_states):

        # find value of each action
        action_values = np.zeros(env.n_actions, dtype=np.float32)
        for a in range(env.n_actions):
            # get next states, compute probabilities and rewards
            S_ = [s_ for s_ in range(env.n_states)]
            P = [env.p(s_, s, a) for s_ in S_]
            R = [env.r(s_, s, a) for s_ in S_]

            # value <- sum_s_[ P(s_ |s, a) * ( R(s_|s, a) + gamma * Vs_) ]
            action_values[a] = sum([p * (r + gamma * V[s_])
                                    for s_, p, r in zip(S_, P, R)])

        # maximise rewards by taking action with largest value
        policy[s] = np.argmax(action_values)

    return policy


def policy_iteration(env: Environment, gamma: float, theta: float, max_iterations: int, policy: np.ndarray = None, total_iterations: int = 0) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    In-place iterative deterministic policy evaluation.
    Interleaves policy evaluation and policy improvement.
    Will run recursively untill policy converges and return optimal policy and state-value function.

    args:
        env: the environment
        gamma: the discount factor
        theta: the convergence threshold
        max_iterations: the maximum number of iterations
        policy: the policy being iterated on during recursion (numpy array with shape: (n_states,)) 
        total_iterations: the running total iterations updated during recuresion

    returns:
        A tuple (policy, V, iter)
        where:
            policy - the optimal policy (numpy array with shape: (n_states,)
            V - the state-value function (numpy array with shape: (n_states,)
            iter - the total number of iterations untill covergence
    """
    # initialise policy if not supplied
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)

    # Interleave policy evaluation and policy improvement
    V, iters = policy_evaluation(env, policy, gamma, theta, max_iterations)
    improved_policy = policy_improvement(env, V, gamma)

    # end recursion if converged (policy unchanged in this step)
    if np.all(policy == improved_policy):
        return policy, V, total_iterations

    # recursive call
    return policy_iteration(env, gamma, theta, max_iterations, improved_policy, total_iterations + iters)


def value_iteration(env: Environment, gamma: float, theta: float, max_iterations: int, V: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    In-place iterative value iteration.
    Will run untill convergance or max iterations and return optimal policy and state-value function.

    Update step:
        Vs <- max_a{ sum_s_[ P(s_ |s, a) * ( R(s_|s, a) + gamma * Vs_) ] }
        where:
            P(s_ |s, a) - probability of transitioning to state s_ after taking action a from state s
            R(s_|s, a) - the reward after taking a and transitioning to s_ 
            gamma - discount factor
            Vs_ - the next state's value

    args:
        env: the environment
        gamma: the discount factor
        theta: the convergence threshold
        max_iterations: the maximum number of iterations
        V: the state-value function (numpy array with shape: (n_states,)

     returns:
        A tuple (policy, V, iter)
        where:
            policy - the optimal policy (numpy array with shape: (n_states,)
            V - the state-value function (numpy array with shape: (n_states,)
            iter - the number of iterations untill covergence
    """
    # initialise value if not provided
    if V is None:
        V = np.zeros(env.n_states, dtype=np.float32)
    else:
        V = np.array(V, dtype=float)

    # Value iteration loop
    for iter in range(max_iterations):
        delta = 0

        # do 1 round of update for each state
        for s in range(env.n_states):
            v = V[s]

            # find value of each action
            action_values = np.zeros(env.n_actions, dtype=np.float32)
            for a in range(env.n_actions):
                # get next states, compute probabilities and rewards
                S_ = [s_ for s_ in range(env.n_states)]
                P = [env.p(s_, s, a) for s_ in S_]
                R = [env.r(s_, s, a) for s_ in S_]

                # value <- sum_s_[ P(s_ |s, a) * ( R(s_|s, a) + gamma * Vs_) ]
                action_values[a] = sum([p * (r + gamma * V[s_])
                                        for s_, p, r in zip(S_, P, R)])

            # value for state is maximum value for all actions
            V[s] = action_values.max()

            delta = max(delta, abs(v - V[s]))

        # check for convergance
        if delta <= theta:
            break

    # build policy from Value function (policy_improvement does this)
    policy = policy_improvement(env, V, gamma)

    return policy, V, iter + 1
