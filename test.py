# Built-in modules
import os
import time

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv

from sb3.argparser import load_config_file
import browser_gym_env

# Existing algorithm implementations
existing_implementations = ["SAC"]

cfg = load_config_file()

test_name = str(time.time()) if cfg.inference.test_name=="auto" else cfg.inference.test_name

# Derive the artefact directory from this
cfg.artefact_path = os.path.join(cfg.inference.artefact_base_path, test_name)
if not os.path.exists(cfg.artefact_path):
    os.makedirs(cfg.artefact_path)

algo = cfg.algorithm_config.algorithm_type

# Check that the active algorithm type has been implemented
assert(algo in existing_implementations)

if algo == "SAC":
    algo_class = SAC

model_load_path = cfg.inference.model_load_path

# Prepare environment
env = make_vec_env(
    cfg.env_config.name,
    n_envs=cfg.env_config.n_envs, 
    vec_env_cls=SubprocVecEnv, 
    env_kwargs={
        "cfg": cfg
    },
    vec_env_kwargs=dict(start_method='fork')
)

# Stack Frames, if enabled
frame_stack_n = cfg.env_config.frame_stack
if frame_stack_n>1:
    env = VecFrameStack(env, n_stack=frame_stack_n)

# Create the model
model = algo_class.load(model_load_path, env=env, device=cfg.algorithm_config.device)

deterministic_pi = cfg.inference.deterministic
log_errors = cfg.inference.log_errors
record_video = cfg.inference.record_video
sleep_duration = cfg.inference.wait_seconds

test_reward = 0

for i in range(cfg.inference.n_episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated = env.step(action)
        test_reward += reward[0]
        time.sleep(sleep_duration)

env.close()

# TODO: Fix video recording bug (onpaint size and wrong location)