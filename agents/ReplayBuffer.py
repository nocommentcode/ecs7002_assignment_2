from collections import deque


class ReplayBuffer:
    def __init__(self, buffer_size, random_state):
        # Replay buffer data structure:
        self.buffer = deque(maxlen=buffer_size)

        self.random_state = random_state

    def __len__(self):
        return len(self.buffer)

    def append(self, transition):
        self.buffer.append(transition)

    def draw(self, batch_size):
        N = len(self.buffer)

        # Randomly sampling `batch_size` buffer indices without replacement:
        indicies = self.random_state.choice(N, size=batch_size, replace=False)

        # return transitions corresponding to above indices
        return [self.buffer[i] for i in indicies]
