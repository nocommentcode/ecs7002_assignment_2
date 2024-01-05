import torch
import numpy as np


class DeepQNetwork(torch.nn.Module):
    def __init__(self, wenv, learning_rate, kernel_size,
                 conv_out_channels, fc_out_features, seed):
        torch.nn.Module.__init__(self)
        torch.manual_seed(seed)

        # Convolutional layer:
        self.conv_layer = torch.nn.Conv2d(in_channels=wenv.state_shape[0],
                                          out_channels=conv_out_channels,
                                          kernel_size=kernel_size, stride=1)

        # h ==> Number of rows in grid, w ==> Number of columns in grid
        h = wenv.state_shape[1] - kernel_size + 1
        w = wenv.state_shape[2] - kernel_size + 1

        # Fully connected layer:
        self.fc_layer = torch.nn.Linear(in_features=h*w*conv_out_channels,
                                        out_features=fc_out_features)

        # Output layer:
        self.output_layer = torch.nn.Linear(in_features=fc_out_features,
                                            out_features=wenv.n_actions)

        # Optimiser for gradient descent:
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        """
        Model forward pass

        args: x - the input states 

        `x.shape` = (B, 4, h, w)
        where:
            B ==> Number of states
            4 ==> Number of channels per state representation
            h ==> Number of rows in the playing grid
            w ==> Number of columns in the playing grid
        """
        # Setting the activation function:
        activation = torch.nn.ReLU()

        # Converting inputted array into a tensor:
        y = torch.tensor(x, dtype=torch.float)

        # Feeding forward the input to convolution layer:
        y = self.conv_layer(y)
        y = activation(y)

        # flatten x on 2nd dimension to keep number of samples unflattened
        y = torch.flatten(y, start_dim=1)

        # Feeding forward the input to fully-connected layer:
        y = self.fc_layer(y)
        y = activation(y)

        # Feeding forward the input to output layer & returning output:
        y = self.output_layer(y)
        return y

    def train_step(self, transitions, gamma, tdqn):
        """
        Training step for model

        args: 
            transitions: batch of (states, actions, rewards, next_states, dones) transitions
            gamma: discount factor
            tdqn: target deep Q-network
        """
        # Organising the transitions data into separate arrays:
        states = np.array([transition[0] for transition in transitions])
        actions = np.array([transition[1] for transition in transitions])
        rewards = np.array([transition[2] for transition in transitions])
        next_states = np.array([transition[3] for transition in transitions])
        dones = np.array([transition[4] for transition in transitions])

        # Obtaining current action-value estimates:
        q = self(states)

        # keeping action-values for previously taken actions:
        q = q.gather(1, torch.Tensor(actions).view(len(transitions), 1).long())
        q = q.view(len(transitions))

        # get target using target_network to stabilise training
        with torch.no_grad():
            # get next action values from target network and set to 0 if episode done
            next_q = tdqn(next_states).max(dim=1)[0] * (1 - dones)

            # one-step rewards given the stored rewards:
            target = torch.Tensor(rewards) + gamma * next_q

        # loss is the mean squared error between q and target
        loss = torch.nn.functional.mse_loss(q, target.to(torch.float32))

        # Performing gradient descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
