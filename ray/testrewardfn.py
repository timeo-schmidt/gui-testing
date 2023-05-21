import gymnasium as gym
import browser_gym_env
import time
# from stable_baselines3.common.env_checker import check_env

#from gymnasium.utils.env_checker import check_env
from ray.rllib.utils import check_env

import readline
import code


env = gym.make('browser_gym_env/WebBrowserEnv-v0')

env.reset()

# variables = globals().copy()
# variables.update(locals())
# shell = code.InteractiveConsole(variables)
# shell.interact()

# Repeatedly click at 0,0
while True:
    env.step([0,0])
    time.sleep(0.5)