import numpy as np


class LinearWrapper:
    """
    A wrapper for environments to encode states as linear features.
    Maps each state action pair into a m dimensional feature vector.
    """

    def __init__(self, env):
        self.env = env
        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        """
        Encodes the state into features

        args: 
            state: integer

        returns:
            features: numpy array shape: (n_actions, n_features) 
        """
        features = np.zeros((self.n_actions, self.n_features))

        # sets the features for this state and all actions to 1
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    def decode_policy(self, theta):
        """
        Decodes the policy to be visible in the environment space

        args:
            theta: the weights

        returns:
            A tuple (policy, value)
        """
        # initialise policy and value
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        # for each state build q table
        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)

            # find policy and value of state using q table
            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)
