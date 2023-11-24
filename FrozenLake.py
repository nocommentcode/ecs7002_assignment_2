
from Environement import Environment
import numpy as np

from utils import _printoptions, play

SMALL_LAKE = [['&', '.', '.', '.'],
              ['.', '#', '.', '#'],
              ['.', '.', '.', '#'],
              ['#', '.', '.', '$']]

BIG_LAKE = [['&', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '#', '.', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '.'],
            ['.', '#', '#', '.', '.', '.', '#', '.'],
            ['.', '#', '.', '.', '#', '.', '#', '.'],
            ['.', '.', '.', '#', '.', '.', '.', '$']]


class FrozenLake(Environment):
    START_SYMBOL = "&"
    FROZEN_SYMBOL = "."
    HOLE_SYMBOL = "#"
    GOAL_SYMBOL = "$"

    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
        lake =  [['&', '.', '.', '.'],
                ['.', '#', '.', '#'],
                ['.', '.', '.', '#'],
                ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)

        self.slip = slip

        n_states = self.lake.size + 1
        n_actions = 4

        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == self.START_SYMBOL)[0]] = 1.0

        self.absorbing_state = n_states - 1
        self.transitions = self.build_transition_probs(n_states, n_actions)

        Environment.__init__(self, n_states, n_actions,
                             max_steps, pi, seed=seed)

    def build_transition_probs(self, n_states, n_actions):
        # state, next state, action
        transitions = np.zeros((n_states, n_states, n_actions))

        # all actions in absorbing state -> absorbing state
        transitions[self.absorbing_state, self.absorbing_state, :] = 1.0

        # all actions in goal -> absorbing state
        transitions[self.absorbing_state, np.where(
            self.lake_flat == self.GOAL_SYMBOL)] = 1.0

        # all actions in hole -> absorbing state
        transitions[self.absorbing_state, np.where(
            self.lake_flat == self.HOLE_SYMBOL)] = 1.0

        # all remaining states
        frozen_states = np.where(self.lake_flat == self.FROZEN_SYMBOL)[0]
        start_states = np.where(self.lake_flat == self.START_SYMBOL)[0]
        states = np.concatenate((frozen_states, start_states))

        n_col, n_rows = self.lake.shape
        col_idx = states % n_col
        row_idx = states // n_rows

        # calculate resulting state when taking each action (ensuring bounds are respected)
        up_states = states - n_col
        up_states[row_idx <= 0] = states[row_idx <= 0]

        left_states = states - 1
        left_states[col_idx <= 0] = states[col_idx <= 0]

        down_states = states + n_col
        down_states[row_idx >= n_rows - 1] = states[row_idx >= n_rows - 1]

        right_states = states + 1
        right_states[col_idx >= n_col - 1] = states[col_idx >= n_col - 1]

        p_slip = self.slip / 4
        p = (1 - self.slip) + p_slip

        # Assigns slip probability to all actions except selected action
        # ie. prob of ending in up state is:
        # - non slip probability if action == up
        # - slip probability if action == left, down or right
        def build_probabilities(action_index: int):
            probs = [
                p if i == action_index else p_slip for i in range(n_actions)]
            return np.array(probs)

        # assign probability to each resulting state from remaining states
        transitions[up_states, states] += build_probabilities(0)
        transitions[left_states, states] += build_probabilities(1)
        transitions[down_states, states] += build_probabilities(2)
        transitions[right_states, states] += build_probabilities(3)

        return transitions

    def step(self, action):
        state, reward, done = Environment.step(self, action)

        done = (state == self.absorbing_state) or done

        return state, reward, done

    def p(self, next_state, state, action):
        return self.transitions[next_state, state, action]

    def is_goal(self, state: int):
        if state == self.absorbing_state:
            return False
        return self.lake_flat[state] == self.GOAL_SYMBOL

    def r(self, next_state, state, action):
        # get reward 1 for taking action in goal
        if self.is_goal(state):
            return 1.0

        # otherwise always 0
        return 0.0

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', '<', '_', '>']

            print('Lake:')
            print(self.lake)

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))


if __name__ == "__main__":
    env = FrozenLake(SMALL_LAKE, slip=0.1, max_steps=16, seed=0)
    play(env)
