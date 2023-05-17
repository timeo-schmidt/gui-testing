import gymnasium as gym

import ray
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.a2c import A2CConfig

from browser_gym_env.envs.web_browser_env import WebBrowserEnv

ray.init()

# env = gym.make("CartPole-v1")

config = (
    A2CConfig()
    .training(grad_clip=30.0)
    .environment(WebBrowserEnv)
    .rollouts(num_rollout_workers=0, num_envs_per_worker=5)
    .resources(num_gpus=0)
)

# Build.
algo = config.build()

for i in range(5000):
    print("TRAINING CYCLE: ", i)
    result = algo.train()
    print(pretty_print(result))

    if i % 100 == 0:
        checkpoint = algo.save()
        print("checkpoint saved at", checkpoint)

algo.stop()