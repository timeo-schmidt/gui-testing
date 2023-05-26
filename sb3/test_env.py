from browser_gym_env.envs.masked_web_browser_env import MaskedWebBrowserEnv
from stable_baselines3.common.env_checker import check_env

env = WebBrowserEnv()
check_env(env)