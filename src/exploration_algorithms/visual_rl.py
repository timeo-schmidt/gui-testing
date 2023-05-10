import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToTensor
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns


from .abstract import AbstractExplorationAlgorithm
from dataclasses import dataclass

"""
Agent Action Classes
"""
class AgentClick:
    def __init__(self, agent, x=None, y=None):
        self.agent = agent
        self.screen_size = agent.app_interface.get_window_size()
        if x is None or y is None:
            self.x = random.randint(0, self.screen_size["width"])
            self.y = random.randint(0, self.screen_size["height"])
            self.is_random = True
        else:
            self.x = int(x*self.screen_size["width"])
            self.y = int(y*self.screen_size["height"])
            self.is_random = False
    def perform_action(self):
        self.agent.app_interface.click(self.x, self.y)
    def get_action_string(self):
        # X, Y and Random
        return f"Click: {self.x}, {self.y} (Random: {self.is_random})"
    def get_action_index(self):
        return 0


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
                 terminal_state_penalty=-1,
                 batch_size=32,
                 buffer_size=100000,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.997,
                 target_update_freq=1000,
                 learning_rate=0.0025,
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

        self.unique_visible_elements = set()
        self.terminal_state_penalty = terminal_state_penalty

        input_shape = (3, self.downscale_size[0], self.downscale_size[1])
        self.policy_net = CNN(input_shape, num_actions, action_parameters).to(self.device)
        self.target_net = CNN(input_shape, num_actions, action_parameters).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss() 

        self.loss_history = []


    def plot_click_locations(self):
        click_x = []
        click_y = []
        click_color = []

        for transition in self.buffer:
            _, action, _, _, _ = transition
            if isinstance(action, AgentClick):
                click_x.append(action.x)
                click_y.append(action.y)
                click_color.append("blue" if action.is_random else "red")

        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=click_x, y=click_y, hue=click_color, palette=["red", "blue"])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Click Locations (Blue: Random, Red: Model-Selected)")
        plt.gca().invert_yaxis()  # Invert the y-axis to match the coordinate system used by app_interface
        plt.savefig("click_locations.png")


    def explore(self, num_steps):

        # Calculate the reward once at the beginning, to initialize the unique_visible_elements set
        _, elements = self.get_state()
        self.calculate_reward(elements)

        step_count = 0

        while step_count < num_steps:
            screenshot, elements = self.get_state()
            action = self.select_action(screenshot)
            action.perform_action()

            # Observe reward and next state
            next_screenshot, next_elements = self.get_state()
            reward = self.calculate_reward(next_elements)

            print("Epoch: {}, Reward: {}, Action: {}, Epsilon: {}".format(step_count, reward, action.get_action_string(), self.epsilon))

            # If the reward is -1, then the agent has reached a terminal state
            terminal_state = not bool(reward)

            # Store transition in replay buffer
            self.buffer.append((screenshot, action, reward, next_screenshot, terminal_state))

            if len(self.buffer) >= self.batch_size:
                self.optimize_model()

            step_count += 1

            # Update the target network
            if step_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict()) 
        
        # After the exploration is finished, produce a plot of all the clicks
        self.plot_click_locations()

    def get_state(self):
        elements = self.app_interface.get_all_elements()

        screenshot = self.app_interface.get_screenshot(size=self.downscale_size)
        screenshot = ToTensor()(screenshot).unsqueeze(0).to(self.device)

        return screenshot, elements

    def select_action(self, screenshot):
        if np.random.rand() > self.epsilon:
            with torch.no_grad():
                action_class_output, action_parameter_output = self.policy_net(screenshot)
                action = action_class_output.max(1)[1].view(1, 1).item()
                action_params = action_parameter_output[action].view(1, -1).tolist()[0]
                if action == 0:
                    return AgentClick(self, action_params[0], action_params[1])
        else:
            random_action = random.randrange(self.num_actions)
            if(random_action==0):
                return AgentClick(self)

    def filter_visible_web_elements(self, web_element_list):
        visible_elements = []
        for element in web_element_list:
            try:
                if element.is_displayed():
                    visible_elements.append(element)
            except:
                pass
        return set(visible_elements)

    def calculate_reward(self, next_elements):

        # Inject overrides if a "terminal" state was reached
        terminal = self.app_interface.fix_deadends()
        if terminal:
            return self.terminal_state_penalty # Penalise for reaching a terminal state

        # Get all the new elements that have not been seen previously in self.unique_visible_elements
        new_elements = set(next_elements) - self.unique_visible_elements
        # Get the visible elements from the new elements
        visible_new_elements = self.filter_visible_web_elements(new_elements)
        # Add the visible new elements to the set of unique visible elements
        self.unique_visible_elements = self.unique_visible_elements.union(visible_new_elements)

        # The reward is the number of new elements that have been seen
        reward = len(visible_new_elements)
        return reward

    def optimize_model(self):

        # self.buffer.append((screenshot, action, reward, next_screenshot, terminal_state))
        # Sample a minibatch from the replay buffer
        minibatch = random.sample(self.buffer, self.batch_size)
        screenshot_batch, action_batch, reward_batch, next_screenshot_batch, terminal_state_batch = zip(*minibatch)

        # Convert the batches to tensors
        screenshot_batch = torch.cat(screenshot_batch).to(self.device)
        action_class_batch = torch.tensor([a.get_action_index() for a in action_batch], dtype=torch.int64).unsqueeze(1).to(self.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_screenshot_batch = torch.cat(next_screenshot_batch).to(self.device)
        terminal_state_batch = torch.tensor(terminal_state_batch, dtype=torch.bool).to(self.device)

        # Compute Q values for the current state-action pairs
        q_values, _ = self.policy_net(screenshot_batch)
        q_values = q_values.gather(1, action_class_batch)

        # Compute the target Q values for the next states using the target network
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_screenshot_batch)
            next_q_values_max = next_q_values.max(1)[0].unsqueeze(1)
            next_q_values_max[terminal_state_batch] = 0.0  # Set Q values to 0 for terminal states
            target_q_values = reward_batch + self.gamma * next_q_values_max

        # Compute the loss and optimize the policy network
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Store the loss
        self.loss_history.append(loss.item())
        # Plot the loss curve
        plt.plot(self.loss_history)
        plt.savefig("loss.png")
        plt.close()
        # Print loss and current progress information
        print("Minibatch Backpropagation Loss: {}".format(loss.item()))

        # Decay epsilon for epsilon-greedy exploration
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)
