import numpy as np

from environment.FrozenLake import FrozenLake


class FrozenLakeImageWrapper:
    """
    A wrapper for the FrozenLake environment which returns states as images.

    The images are 4 channel images:
        1) position of the agent
        2) position of the holes
        3) position of the frozen tiles
        4) position of the goal

    """

    def __init__(self, env: FrozenLake):
        self.env = env
        lake = self.env.lake
        self.n_actions = self.env.n_actions
        self.state_shape = (4, lake.shape[0], lake.shape[1])
        self.state_image = self.build_state_images(lake)

    def build_state_images(self, lake):
        """
        Builds the state images for all states
        """
        # dimension 2, 3 and 4
        lake_image = [(lake == c).astype(float) for c in [
            self.env.START_SYMBOL, self.env.GOAL_SYMBOL, self.env.HOLE_SYMBOL]]

        state_image = {}

        # build image for each state including absorbing state
        for state in range(lake.size + 1):
            rows, cols = lake.shape

            # 1st dimension
            agent_image = np.zeros((rows, cols))

            # if not absorbing state set player position to 1
            if state != self.env.absorbing_state:
                agent_image[state % rows, state // cols] = 1.0

            # state image is all 4 dimensions
            state_image[state] = np.stack([agent_image] + lake_image)

        return state_image

    def encode_state(self, state):
        """
        Encodes the state to an image
        """
        return self.state_image[state]

    def decode_policy(self, dqn):
        """
        Decodes the policy so it can be viewed
        """

        states = np.array([self.encode_state(s)
                          for s in range(self.env.n_states)])
        # torch.no_grad omitted to avoid import
        q = dqn(states).detach().numpy()

        policy = q.argmax(axis=1)
        value = q.max(axis=1)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)
