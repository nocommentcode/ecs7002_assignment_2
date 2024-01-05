## ECS7002P Assigment 2 repo

### Contributors: Malo Hamon, Pranav Gopalkrishna


### Package Structure:

- **agents:** contains all the rl agent algorithms and the classes necessary
    - <u>tabular_model_rl:</u> Policy Iteration and Value iteration algorithms
    - <u>tabular_model_free_rl:</u> SARSA Control and Q-Learning Control algorithms
    - <u>non_tabular_model_free_rl</u>: Linear SARSA Control, Linear Q-Learning Control and Deep Q Learning algorithms
    - <u>DeepQNetwork:</u> The Q-Network Neural Network class
    - <u>ReplayBuffer:</u> The replay buffer for DQL

- **environment:** contains all the files related to the environment.
    - <u>EnvironmentModel:</u> Base class for environments
    - <u>Environment:</u> Abstract class for an environment
    - <u>FrozenLake:</u> Class for the Frozen Lake environment  
    - <u>LinearWrapper:</u> A Wrapper for the Frozen lake environment used for linear sarsa and q learning  
    - <u>FrozenLakeImageWrapper:</u>A Wrapper for the Frozen lake environment used for Deep Q Learning  
    - <u>EpisodeRewardsWrapper:</u> A Wrapper for an environment that keeps track of all rewards obtained during each episode of training

###  Main files:
- **tests:** runs tests for the environment implementation
- **main:** main function that runs all agents and prints resulting policy and value
- **count_tabular_model_iterations:** script used for question 1 of the report, logs the number of iterations taken for policy and value iteration to converge