# Algorithm imports
from stable_baselines3 import A2C
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

# Environment imports
import browser_gym_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Callback imports
from stable_baselines3.common.callbacks import CheckpointCallback

# Experiment parameters
EXPERIMENT_NAME = "a2c_vanilla_atari_params"
MODEL_SAVE_PATH = "./models/"
N_ENVS = 10

# Prepare environment
env = make_vec_env(
    "browser_gym_env/WebBrowserEnv-v0", 
    n_envs=N_ENVS, 
    vec_env_cls=SubprocVecEnv, 
    env_kwargs={"masking": False, "log_steps": False, "grayscale":True},
    vec_env_kwargs=dict(start_method='fork')
)

env = VecFrameStack(env, n_stack=5)

# Prepare model
model = A2C(
    "CnnPolicy", 
    env,
    verbose=1, 
    device="mps",
    seed=42,
    tensorboard_log="./tensorboard/",
    ent_coef=0.01,
    vf_coef=0.5,
    # policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)),
    policy_kwargs=dict(optimizer_kwargs=dict(amsgrad=True)),
    learning_rate=1e-4,
    gamma=0.99
)

# Train the model and save checkpoints
checkpoint_callback = CheckpointCallback(
  save_freq=10000,
  save_path=MODEL_SAVE_PATH,
  name_prefix=EXPERIMENT_NAME,
  save_replay_buffer=True,
  save_vecnormalize=True,
)

model.learn(
    total_timesteps=500000, 
    log_interval=4, 
    tb_log_name=EXPERIMENT_NAME, 
    callback=checkpoint_callback
)