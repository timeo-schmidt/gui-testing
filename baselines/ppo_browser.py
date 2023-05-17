import gym

import browser_gym_env

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

env = gym.make('browser_gym_env/WebBrowserEnv-v0')

model = SAC("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
# model.save("ppo_cartpole")