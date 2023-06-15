"""

This is a baseline implementation for tabular Q learning based on the publication:

@inproceedings{Adamo,
    title = {{Reinforcement learning for Android GUI testing}},
    year = {2018},
    booktitle = {Proceedings of the 9th ACM SIGSOFT International Workshop on Automating TEST Case Design, Selection, and Evaluation},
    author = {Adamo, David and Khan, Md Khorrom and Koppula, Sreedevi and Bryce, Renée},
    month = {11},
    pages = {2--8},
    publisher = {ACM},
    address = {New York, NY, USA},
    isbn = {9781450360531},
    doi = {10.1145/3278186.3278187}
}

The implementation is re-created based on the description in the paper and adapted to
be compatible with the web browser environment.


"""

import random
import math

Q_table = {}            # Q table containing tuples (times_executed, q_value)

Q_INIT = 500            # Initial value of q
Q_DISCOUNT = 0.9        # How much to multiply the q value by when updating

# Store last action taken (initialized as None)
last_action = None
number_actions_previously = None

def resetQtable():
    global last_action
    global number_actions_previously
    global Q_table
    
    Q_table = {}
    last_action = None
    number_actions_previously = None

def get_actions_from_xpath(env, xpath):
    # Use the Selenium driver to find the element by xpath
    x,y = env.web_app_interface.get_element_center(xpath)

    # Convert the action values to -1, 1 range by scaling according to env.viewport_size
    x = (x / env.viewport_size[0]) * 2 - 1
    y = (y / env.viewport_size[1]) * 2 - 1

    # Return the x and y action values
    return x, y

def get_action(obs, env, prev_reward=0):
    global last_action
    global number_actions_previously
    global Q_table

    # Update Q value for last action, if it was taken
    if last_action is not None:
        # gamma is calculated as 0.9*e^(-0.1*(n-1)) where n is the number of actions taken
        gamma = 0.9 * math.exp(-0.1 * (number_actions_previously - 1))
        Q_table[last_action] = (Q_table[last_action][0]+1 , 1/(Q_table[last_action][0]+1) + gamma * prev_reward)

    # Get all the interactable and visible xpaths from the environment
    els = env.get_interactable_and_visible_xpaths()

    # Remove duplicates
    els = list(set(els))

    # Add unseen elements to the Q-table with initial value
    for el in els:
        if el not in Q_table:
            Q_table[el] = (0, Q_INIT)

    # Create a list of all the Q-values of the available actions
    q_values = [Q_table[el][1] for el in els]

    # Compute the total of all Q-values
    total_q = sum(q_values)

    # Create a list of probabilities for each action
    probabilities = [q_value / total_q for q_value in q_values]

    # Randomly choose an action from the available actions, based on the probabilities
    chosen_action_xpath = random.choices(els, probabilities)[0]

    # Update the last action taken
    last_action = chosen_action_xpath

    # Update the number of actions taken
    number_actions_previously = len(q_values)

    # Get the (x, y) coordinates of the chosen action's XPath
    chosen_action_coordinates = get_actions_from_xpath(env, chosen_action_xpath)

    return chosen_action_coordinates
