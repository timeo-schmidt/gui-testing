import gymnasium as gym
import browser_gym_env
# from stable_baselines3.common.env_checker import check_env

#from gymnasium.utils.env_checker import check_env
from ray.rllib.utils import check_env


env = gym.make('browser_gym_env/WebBrowserEnv-v0')
# env = gym.make("CarRacing-v2")
print(check_env(env))