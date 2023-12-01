import numpy as np


class FrozenLakeImageWrapper:
    """
    A wrapper for the FrozenLake environment which returns states as images.

    The images are 4 channel images:
        1) position of the agent
        2) position of the holes
        3) position of the frozen tiles
        4) position of the goal

    """

    def __init__(self, env):
        self.env = env

        lake = self.env.lake

        self.n_actions = self.env.n_actions
        self.state_shape = (4, lake.shape[0], lake.shape[1])

        lake_image = [(lake == c).astype(float) for c in ['&', '#', '$']]

        self.state_image = {self.env.absorbing_state:
                            np.stack([np.zeros(lake.shape)] + lake_image)}
        for state in range(lake.size):
            rows, cols = lake.shape
            position = np.zeros((rows, cols))

            position[state % rows, state // cols] = 1.0
            self.state_image[state] = np.stack([position] + lake_image)

    def encode_state(self, state):
        return self.state_image[state]

    def decode_policy(self, dqn):
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
