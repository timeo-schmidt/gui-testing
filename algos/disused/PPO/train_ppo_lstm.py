# Algorithm imports
from sb3_contrib import RecurrentPPO

# Environment imports
import browser_gym_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Callback imports
from stable_baselines3.common.callbacks import CheckpointCallback

# Experiment parameters
EXPERIMENT_NAME = "ppo_lstm_squashed_500k"
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

# env = VecFrameStack(env, n_stack=4)

# Prepare model
model = RecurrentPPO(
    "CnnLstmPolicy", 
    env,
    verbose=1, 
    device="mps", 
    tensorboard_log="./tensorboard/",
    n_steps=20,
    n_epochs=4,
    batch_size=200,
    # learning rate lin_2.5e-4 to callable
    learning_rate=lambda f: f * 2.5e-4,
    # clip range lin_0.1 to callable
    clip_range=lambda f: f * 0.1,
    vf_coef=0.5,
    ent_coef=0.01,
    policy_kwargs={"squash_output":True},
)

# Train the model and save checkpoints
checkpoint_callback = CheckpointCallback(
  save_freq=5000,
  save_path=MODEL_SAVE_PATH,
  name_prefix=EXPERIMENT_NAME,
  save_replay_buffer=True,
  save_vecnormalize=True,
)

model.learn(
    total_timesteps=1000000, 
    log_interval=4, 
    tb_log_name=EXPERIMENT_NAME, 
    callback=checkpoint_callback
)