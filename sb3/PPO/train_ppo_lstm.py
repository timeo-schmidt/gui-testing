# Algorithm imports
from stable_baselines3 import PPO

# Environment imports
import browser_gym_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Callback imports
from stable_baselines3.common.callbacks import CheckpointCallback

# Experiment parameters
EXPERIMENT_NAME = "ppo_vanilla_100k"
MODEL_SAVE_PATH = "./models/"
N_ENVS = 10
MAX_BUFFER_SIZE = 30000


# Prepare environment
env = make_vec_env(
    "browser_gym_env/WebBrowserEnv-v0", 
    n_envs=N_ENVS, 
    vec_env_cls=SubprocVecEnv, 
    env_kwargs={"masking": False, "log_steps": True},
    vec_env_kwargs=dict(start_method='fork')
)

env = VecFrameStack(env, n_stack=4)

# Prepare model
model = PPO(
    "CnnPolicy", 
    env,
    verbose=1, 
    device="mps", 
    tensorboard_log="./tensorboard/",
    n_steps=10
)


# Train the model and save checkpoints
checkpoint_callback = CheckpointCallback(
  save_freq=1000,
  save_path=MODEL_SAVE_PATH,
  name_prefix=EXPERIMENT_NAME,
  save_replay_buffer=True,
  save_vecnormalize=True,
)

model.learn(
    total_timesteps=100000, 
    log_interval=4, 
    tb_log_name=EXPERIMENT_NAME, 
    callback=checkpoint_callback
)