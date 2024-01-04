import numpy as np
from environment.FrozenLake import FrozenLake
from environment.FrozenLake import BIG_LAKE
from agents.tabular_model_rl import policy_iteration, value_iteration
import sys


def main():
    SEED = 0
    GAMMA = 0.9

    lake = BIG_LAKE
    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=SEED)

    # policy iteration with infinite iterations and threshold of 0 will return optimal policy
    print("Running Policy iteration on big lake...")
    policy, value, iterations = policy_iteration(
        env, GAMMA, theta=0, max_iterations=sys.maxsize)
    env.render(policy, value)
    print(f"Optimal policy found after {iterations} iterations")

    print("\n\n")

    # value iteration with infinite iterations and threshold of 0 will return optimal policy
    print("Running Value iteration on big lake...")
    policy, value, iterations = value_iteration(
        env, GAMMA, theta=0, max_iterations=sys.maxsize)
    env.render(policy, value)
    print(f"Optimal policy found after {iterations} iterations")


if __name__ == "__main__":
    main()
