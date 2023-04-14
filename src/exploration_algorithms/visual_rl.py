import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from collections import deque

from .abstract import AbstractExplorationAlgorithm

"""
CNN class that takes in the input shape and number of actions

Input params:
    input_shape = (3, 480, 640)
    num_actions = 4
    action_parameters = [0, 0, 2, 2] # 0 means no parameter, 2 means 2 parameters (x and y)

Output parameters example:
    action_class_output = [0.1, 0.2, 0.3, 0.4] # 4 actions
    action_outputs = [None, None, [0.1, 0.2], [0.3, 0.4]] # 4 actions, 2 parameters each
"""
class CNN(nn.Module):
    def __init__(self, input_shape, num_actions, action_parameters):

        super(CNN, self).__init__()

        # Using three convolutional layers
        self.conv1 = nn.Sequential(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU())

        # Dynamically calculate the input dimension for the fully connected layer
        self.fc_input_dim = self._get_fc_input_dim(input_shape)
        

        # Action classifier
        self.action_classifier = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
            nn.Sigmoid()
        )

        # Action parameter heads
        self.action_heads = nn.ModuleList()

        for parameters in action_parameters:
            head = nn.Sequential(
                nn.Linear(self.fc_input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, parameters) if parameters > 0 else None,
                nn.Sigmoid()
            )
            self.action_heads.append(head)


    # Function to dynamically calculate the input dimension for the fully connected layer
    def _get_fc_input_dim(self, input_shape):
        x = torch.zeros(input_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(np.prod(x.size()))

    # Forward pass
    def forward(self, x):

        # Pass the input through the convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1) # Flatten the output of the convolutional layers
        
        # Action Classifier Head
        action_class_output = self.action_classifier(x)
        
        # Action Parameter Heads
        action_outputs = []
        for head in self.action_heads:
            if head is not None:
                action_outputs.append(head(x))
            else:
                action_outputs.append(None)
        
        # Return the action class output and action parameter outputs
        return action_class_output, action_outputs


"""
This is the class that implements the Visual Reinforcement Learning Exploration Algorithm
It is based on Double DQN and relies on a CNN with multiple heads to predict the action class and action parameters
"""
class VisualRL(AbstractExplorationAlgorithm):
    """
    Initializes the VisualRL class with given parameters.

        :param app_interface: An instance of a class that inherits from AbstractAppInterface.
        :param batch_size: The size of the batch used for experience replay. Default is 32.
        :param buffer_size: The maximum size of the replay buffer. Default is 100000.
        :param gamma: The discount factor used in the Q-learning update. Default is 0.99.
        :param epsilon_start: The initial value of epsilon for the epsilon-greedy exploration. Default is 1.0.
        :param epsilon_end: The final value of epsilon for the epsilon-greedy exploration. Default is 0.01.
        :param epsilon_decay: The decay rate of epsilon for each step. Default is 0.995.
        :param target_update_freq: The frequency (number of steps) at which the target network is updated. Default is 1000.
        :param learning_rate: The learning rate for the optimizer. Default is 0.00025.
        :param device: The device to use for computation ('cpu' or 'cuda'). If None, it will use 'cuda' if available, otherwise 'cpu'. Default is None.
    """
    def __init__(self,
                 app_interface,
                 num_actions=1,
                 action_parameters=[2],
                 batch_size=1,
                 buffer_size=100000,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.9999,
                 target_update_freq=1000,
                 learning_rate=0.00025,
                 downscale_size=(256, 256),
                 device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.learning_rate = learning_rate

        self.app_interface = app_interface
        self.screen_size = self.app_interface.get_window_size()
        self.downscale_size = downscale_size
        self.num_actions = num_actions
        self.action_parameters = action_parameters

        input_shape = (3, self.downscale_size[0], self.downscale_size[1])
        self.policy_net = CNN(input_shape, num_actions, action_parameters).to(self.device)
        self.target_net = CNN(input_shape, num_actions, action_parameters).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss() 


    def explore(self, num_steps):
        step_count = 0

        while step_count < num_steps:
            state = self.get_state()
            action, action_params = self.select_action(state)
            # Execute action
            self.perform_action(action, action_params)

            # Observe reward and next state
            next_state = self.get_state()
            reward, is_terminal = self.calculate_reward(state, next_state)

            # Store transition in replay buffer
            self.buffer.append((state, action, reward, next_state, is_terminal))

            if len(self.buffer) >= self.batch_size:
                self.optimize_model()

            state = next_state
            step_count += 1

            # Update the target network
            if step_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_state(self):
        screenshot = self.app_interface.get_screenshot(size=self.downscale_size)
        state = ToTensor()(screenshot).unsqueeze(0).to(self.device)
        return state

    def select_action(self, state):
        print("Epsilon: {}".format(self.epsilon))
        print(np.random.rand())
        if np.random.rand() > self.epsilon:
            print("Selecting action using policy network")
            with torch.no_grad():
                action_class_output, action_parameter_output = self.policy_net(state)
                action = action_class_output.max(1)[1].view(1, 1).item()
                action_params = action_parameter_output[action].view(1, -1).tolist()[0]
                return action, action_params
        else:
            print("Selecting random action")
            random_action = random.randrange(self.num_actions)
            return random_action, None

    def perform_action(self, action, action_params):
        if action == 0:  # Click
            # Get the x and y coordinates from the action parameters
            x = action_params[0] if action_params is not None else np.random.rand()
            x = int(x * self.screen_size["width"])
            y = action_params[1] if action_params is not None else np.random.rand()
            y = int(y * self.screen_size["height"])
            self.app_interface.click(x, y)

    def calculate_reward(self, state, next_state):
        # Calculate the Mean Squared Error (MSE) between the two states
        mse_loss = nn.MSELoss(reduction='mean')
        visual_difference = mse_loss(state, next_state).item()

        return visual_difference, False

    def optimize_model(self):
        # Sample a minibatch from the replay buffer
        minibatch = random.sample(self.buffer, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*minibatch)

        state_batch = torch.cat(state_batch).to(self.device)
        action_batch = torch.tensor(action_batch, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_state_batch = torch.cat(next_state_batch).to(self.device)
        terminal_batch = torch.tensor(terminal_batch, dtype=torch.bool).to(self.device)

        # Compute the action values for the current state using the policy network
        state_action_values, _ = self.policy_net(state_batch)
        state_action_values = state_action_values.gather(1, action_batch)

        # Compute the action values for the next state using the target network
        with torch.no_grad():
            next_state_values, _ = self.target_net(next_state_batch)
            next_state_actions = next_state_values.max(1, keepdim=True)[1]
            next_state_values = self.policy_net(next_state_batch)[0].gather(1, next_state_actions)
            next_state_values[terminal_batch] = 0.0
            target_values = reward_batch + self.gamma * next_state_values

        # Compute the loss
        loss = self.loss_fn(state_action_values, target_values)

        # Perform the optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the epsilon value for epsilon-greedy exploration
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)