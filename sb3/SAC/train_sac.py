# Algorithm imports
from stable_baselines3 import SAC

# Environment imports
import browser_gym_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Callback imports
from stable_baselines3.common.callbacks import CheckpointCallback

# Experiment parameters
EXPERIMENT_NAME = "sac_graystack_3_500k"
MODEL_SAVE_PATH = "./models/"
N_ENVS = 10
MAX_BUFFER_SIZE = 100000
USE_CHECKPOINTS = None
USE_REPLAY_BUFFER = None

# Prepare environment
env = make_vec_env(
    "browser_gym_env/WebBrowserEnv-v0", 
    n_envs=N_ENVS, 
    vec_env_cls=SubprocVecEnv, 
    env_kwargs={"masking": False, "log_steps": True, "grayscale":True},
    vec_env_kwargs=dict(start_method='fork')
)

env = VecFrameStack(env, n_stack=3)

# Prepare model
model = SAC(
    "CnnPolicy", 
    env,
    verbose=1, 
    device="mps", 
    seed=42,
    tensorboard_log="./tensorboard/",
    # buffer_size=MAX_BUFFER_SIZE,
)

# Load checkpoints
if USE_CHECKPOINTS:
    model = SAC.load(USE_CHECKPOINTS, env=env, device="mps")

# Load replay buffer
if USE_REPLAY_BUFFER:
    model.load_replay_buffer(USE_REPLAY_BUFFER)

# Train the model and save checkpoints
checkpoint_callback = CheckpointCallback(
  save_freq=10000,
  save_path=MODEL_SAVE_PATH,
  name_prefix=EXPERIMENT_NAME,
  save_replay_buffer=False,
  save_vecnormalize=True,
)

model.learn(
    total_timesteps=500000, 
    log_interval=4, 
    tb_log_name=EXPERIMENT_NAME, 
    callback=checkpoint_callback
)