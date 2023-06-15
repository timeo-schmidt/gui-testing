"""
This script runs one of the existing baselines
"""

# Built-in modules
import os
import time
import numpy as np
import gymnasium as gym

from utils.argparser import load_config_file
import browser_gym_env

# Existing algorithm implementations
existing_implementations = ["random", "human", "tabularQ"]

cfg = load_config_file()
cfg.mode = "run_baseline"

test_name = str(time.time()) if cfg.inference.test_name=="auto" else cfg.inference.test_name

# Derive the artefact directory from this
cfg.artefact_path = os.path.join(cfg.inference.artefact_base_path, test_name)
if not os.path.exists(cfg.artefact_path):
    os.makedirs(cfg.artefact_path)

baseline = cfg.baseline.type

# Check that the active algorithm type has been implemented
assert(baseline in existing_implementations)

if baseline == "random":
    from baselines.random_baseline import get_action
elif baseline == "human":
    from baselines.human_baseline import get_action
elif baseline == "tabularQ":
    from baselines.tabularQ_baseline import get_action

# Prepare environment
env = gym.make("browser_gym_env/WebBrowserEnv-v0",cfg=cfg)

reward_list = []
for n in range(cfg.baseline.n_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0
    reward = 0
    while not done:
        if baseline=="tabularQ":
            action = get_action(obs, env, reward)
        action = get_action(obs, env)
        obs, reward, done, _, _ = env.step(action)
        episode_reward += reward
    reward_list.append(episode_reward)

# Produce a results summary
print("Average reward over {} episodes: {}".format(cfg.baseline.n_episodes, sum(reward_list)/len(reward_list)))
# Standard Deviation
print("Standard Deviation: {}".format(np.std(reward_list)))
# Lowest rewards
print("Lowest rewards: {}".format(sorted(reward_list)[:10]))
# Highest rewards
print("Highest rewards: {}".format(sorted(reward_list)[-10:]))

env.close()