
from typing import List, Optional
from environment.Environement import Environment
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
    """
    Frozen lake environemt.
        Squares in the lake can be:
            & - the starting square
            . - lake square
            # - hole square (-1 reward)
            $ - goal square

        actions:
            0 (up) 
            1 (left)
            2 (down)
            3 (right)
    """
    START_SYMBOL = "&"
    FROZEN_SYMBOL = "."
    HOLE_SYMBOL = "#"
    GOAL_SYMBOL = "$"

    UP_ACTION_INDEX = 0
    LEFT_ACTION_INDEX = 1
    DOWN_ACTION_INDEX = 2
    RIGHT_ACTION_INDEX = 3

    def __init__(self, lake: List[List[str]], slip: float, max_steps: int, seed: Optional[int] = None) -> None:
        """
        args:

            lake: A matrix that represents the lake. For example:
                lake =  [['&', '.', '.', '.'],
                        ['.', '#', '.', '#'],
                        ['.', '.', '.', '#'],
                        ['#', '.', '.', '$']]
            slip: The probability that the agent will slip
            max_steps: The maximum number of time steps in an episode
            seed: A seed to control the random number generator (optional)
        """
        # store lake
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)

        n_states = self.lake.size + 1
        n_actions = 4

        # absorbing state will store the player upon making a move in the goal state
        self.absorbing_state = n_states - 1

        # prob distribution over initial states
        pi = np.zeros(n_states, dtype=float)
        # only state with start symbol will be starting state
        pi[np.where(self.lake_flat == self.START_SYMBOL)[0]] = 1.0

        # store slip probability
        self.slip = slip

        # build transition probablities
        self.transitions = self.build_transition_probs(n_states, n_actions)

        Environment.__init__(self, n_states, n_actions,
                             max_steps, pi, seed=seed)

    def build_transition_probs(self, n_states, n_actions):
        """
        Builds an array storing transition probabilities from a state to the next state given the action.
        Will take into account the slip probability

        args:
            n_states: the number of states
            n_actions: the number of allowable actions

        returns:
            transition_probabilities (numpy array with shape: (n_states, n_states, n_actions))
        """
        # transition probability (next state, state, action)
        transitions = np.zeros((n_states, n_states, n_actions))

        # CASE 1: Current state is absorbing state
        # actions in the absorbing state always result in the absorbing state
        # so all actions for abs_state -> abs_state transition will have probability 1
        transitions[self.absorbing_state, self.absorbing_state, :] = 1.0

        # CASE 2: Current state is goal
        # taking any action in the goal will move the player to the absorbing state
        # so all actions for goal_state -> abs_state transition will have probability 1
        transitions[self.absorbing_state, np.where(
            self.lake_flat == self.GOAL_SYMBOL)] = 1.0

        # CASE 3: Current state is a hole
        # taking any action in a holde moves the player to the absorbing state
        # so all actions for hole_state -> abs_state transition will have probability 1
        transitions[self.absorbing_state, np.where(
            self.lake_flat == self.HOLE_SYMBOL)] = 1.0

        # CASE 4: all remaining states (start state or frozen states)
        frozen_states = np.where(self.lake_flat == self.FROZEN_SYMBOL)[0]
        start_states = np.where(self.lake_flat == self.START_SYMBOL)[0]
        remaining_states = np.concatenate((frozen_states, start_states))

        # for the remaining states actions have different probs of transitioning to different states
        # we calculate these probs by:
        #   for each action (up, left, down, right):
        #       1) find the resulting state from all remaining states if this action is taken there
        #       2) ensure this resulting state is within bounds, else resulting state is the origin state
        #       3) assign these resulting states the probability of not slipping for the action taken
        #       4) assign these resulting states the probability of slipping for all other actions

        def assign_probabilities(resulting_states, action_index: int):
            """
            Assigns probabilities from remaining state to resulting states
            (Performs step 3 and 4 described above)
            action_index: set not slipping probability
            all other action indexes: set slipping probability

            args:
                resulting_states: states reached from remaining_states after action performed
                action_index: the index to the action to set non slipping probability to
            """
            # step 4 probability - probability of slipping is slip prob / 4
            # (could have slipped to any of the 4 actions)
            p_slip = self.slip / 4

            # step 3 probability - probability of not slipping
            # (adding on prob of slipping since we could slip into our original action too)
            p = (1 - self.slip) + p_slip

            probs = [
                p if i == action_index else p_slip for i in range(n_actions)]
            probs = np.array(probs)

            # set probabilities for reaching resulting states from remaining_states
            transitions[resulting_states, remaining_states] += probs

        n_col, n_rows = self.lake.shape

        # find collumn and row index of all remaining states
        col_idx = remaining_states % n_col
        row_idx = remaining_states // n_rows

        # UP ACTION
        # step 1:
        up_states = remaining_states - n_col
        # step 2:
        up_states[row_idx <= 0] = remaining_states[row_idx <= 0]
        # steps 3 and 4:
        assign_probabilities(up_states, self.UP_ACTION_INDEX)

        # LEFT ACTION
        # step 1:
        left_states = remaining_states - 1
        # step 2:
        left_states[col_idx <= 0] = remaining_states[col_idx <= 0]
        # steps 3 and 4:
        assign_probabilities(left_states, self.LEFT_ACTION_INDEX)

        # DOWN ACTIONS
        # step 1:
        down_states = remaining_states + n_col
        # step 2:
        down_states[row_idx >= n_rows -
                    1] = remaining_states[row_idx >= n_rows - 1]
        # steps 3 and 4:
        assign_probabilities(down_states, self.DOWN_ACTION_INDEX)

        # RIGHT ACTIONS
        # step 1:
        right_states = remaining_states + 1
        # step 2:
        right_states[col_idx >= n_col -
                     1] = remaining_states[col_idx >= n_col - 1]
        # steps 3 and 4:
        assign_probabilities(right_states, self.RIGHT_ACTION_INDEX)

        return transitions

    def step(self, action):
        """
        Take a step in the environment.

        args:
            action: action

        returns:
            next state, reward, done
        """
        state, reward, done = Environment.step(self, action)

        # override done
        done = (state == self.absorbing_state) or done

        return state, reward, done

    def p(self, next_state, state, action):
        """
        Transition probability from state to next_state when taking action.

        args:
            next_state: next state
            state: current state
            action: action

        returns:
            transition probability
        """
        # return pre computed probability
        return self.transitions[next_state, state, action]

    def is_goal(self, state: int):
        if state == self.absorbing_state:
            return False
        return self.lake_flat[state] == self.GOAL_SYMBOL

    def r(self, next_state, state, action):
        """
        Reward for transitioning from state to next_state when taking action.

        args:
            next_state: next state
            state: current state
            action: action

        returns:
            reward
        """

        # get reward 1 for taking action in goal
        if self.is_goal(state):
            return 1.0

        # otherwise always 0
        return 0.0

    def render(self, policy=None, value=None):
        """
        Render the environment given a policy and value function.
        Sub classes should implement this method.

        args:
            policy: policy to render
            value: value-function to render
        """
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
