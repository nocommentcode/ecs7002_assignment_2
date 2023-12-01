from typing import Optional
import numpy as np

from environment.EnvironmentModel import EnvironmentModel


class Environment(EnvironmentModel):
    """
    An environment which resets to a particular state given a distribution of starting states.
    Subclasses must implement the p and r methods.
    """

    def __init__(self, n_states: int, n_actions: int, max_steps: int, pi: Optional[np.ndarray], seed: Optional[int] = None) -> None:
        """
        args:
            n_states: number of states
            n_actions: number of actions
            max_steps: maximum number of steps in an episode
            pi: distribution of starting states
            seed: random seed
        """
        EnvironmentModel.__init__(self, n_states, n_actions, seed)

        self.max_steps = max_steps

        # initialise starting state distribution to uniform if not provided
        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1./n_states)

    def reset(self) -> int:
        """
        Reset the environment to a starting state.

        returns:
            starting state
        """
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)

        return self.state

    def step(self, action: int) -> (int, float, bool):
        """
        Take a step in the environment.

        args:
            action: action

        returns:
            next state, reward, done
        """
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)

        self.state, reward = self.draw(self.state, action)

        return self.state, reward, done

    def render(self, policy=None, value=None) -> None:
        """
        Render the environment given a policy and value function.
        Sub classes should implement this method.

        args:
            policy: policy to render
            value: value-function to render
        """
        raise NotImplementedError()
