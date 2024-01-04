import numpy as np
from environment.FrozenLake import FrozenLake
from environment.FrozenLake import BIG_LAKE
from agents.tabular_model_rl import policy_iteration, value_iteration

SEED = 0
GAMMA = 0.9
MAX_ITER = 10**100
THETA = 0


def main():
    lake = BIG_LAKE
    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=SEED)

    print("Running Policy iteration on big lake...")
    policy, value, iterations = policy_iteration(
        env, GAMMA, theta=THETA, max_iterations=MAX_ITER)
    env.render(policy, value)

    print(f"Optimal policy found after {iterations} iterations")

    print("\n")
    print("Running Value iteration on big lake...")
    policy, value, iterations = value_iteration(
        env, GAMMA, theta=THETA, max_iterations=MAX_ITER)
    env.render(policy, value)
    print(f"Optimal policy found after {iterations} iterations")


if __name__ == "__main__":
    main()
