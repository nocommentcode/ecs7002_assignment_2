import numpy as np

from environment.FrozenLake import FrozenLake, SMALL_LAKE


def test_frozen_lake_env_probs():
    env = FrozenLake(SMALL_LAKE, slip=0.1, max_steps=16, seed=0)
    actual_transitions = np.load("p.npy")
    predicted_transitions = env.transitions

    assert (actual_transitions == predicted_transitions).sum(
    ) == actual_transitions.size


if __name__ == "__main__":
    test_frozen_lake_env_probs()
    print("All test passed.")
