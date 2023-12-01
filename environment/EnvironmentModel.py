from typing import Tuple
import numpy as np


class EnvironmentModel:
    """
    Base class for environments.

    Environments will call the draw method to get next state and reward from the current state and action.
    Subclasses need to implement the p and r methods.

    States and actions are represented by integers.
    """

    def __init__(self, n_states: int, n_actions: int, seed=None) -> None:
        self.n_states = n_states
        self.n_actions = n_actions

        self.random_state = np.random.RandomState(seed)

    def p(self, next_state: int, state: int, action: int) -> float:
        """
        Transition probability from state to next_state when taking action.

        args:
            next_state: next state
            state: current state
            action: action

        returns:
            transition probability
        """
        raise NotImplementedError()

    def r(self, next_state: int, state: int, action: int) -> float:
        """
        Reward for transitioning from state to next_state when taking action.

        args:
            next_state: next state
            state: current state
            action: action

        returns:
            reward
        """
        raise NotImplementedError()

    def draw(self, state: int, action: int) -> (int, float):
        """
        Draw next state and reward from current state and action.

        args:
            state: current state
            action: action

        returns:
            next state, reward
        """
        p = [self.p(ns, state, action)
             for ns in range(self.n_states)]

        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)

        return next_state, reward
