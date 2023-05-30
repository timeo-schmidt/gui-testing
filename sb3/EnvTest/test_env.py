from browser_gym_env.envs.web_browser_env import WebBrowserEnv
from stable_baselines3.common.env_checker import check_env

env = WebBrowserEnv(masking=False, grayscale=True)
check_env(env)