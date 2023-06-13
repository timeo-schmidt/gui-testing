# Built-in modules
import os

# Third-party libraries
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from sb3.argparser import load_config_file

# Local/application specific modules
import browser_gym_env

# Existing algorithm implementations
existing_implementations = ["SAC"]

cfg = load_config_file()

print(cfg)

# Automatically derive an experiment name
if "auto" in cfg.algorithm_config.experiment_name:
    # Generate the experiment name based on the algorithm type, framestack number, grayscale, masking
    auto_name = f"{cfg.algorithm_config.algorithm_type}_framestack_{cfg.env_config.frame_stack}_"
    auto_name += "grayscale_" if cfg.env_config.grayscale else "color_"
    auto_name += "masked" if cfg.env_config.masking else "unmasked"
    
    # Replace "auto" with the generated name
    cfg.algorithm_config.experiment_name = cfg.algorithm_config.experiment_name.replace("auto", auto_name)

# Derive the artefact directory from this
cfg.artefact_path = os.path.join(cfg.algorithm_config.artefact_base_path, cfg.algorithm_config.experiment_name)
if not os.path.exists(cfg.artefact_path):
    os.makedirs(cfg.artefact_path)

algo = cfg.algorithm_config.algorithm_type

# Check that the active algorithm type has been implemented
assert(algo in existing_implementations)

if algo == "SAC":
    from sb3.SAC.train_sac import create_model

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
model  = create_model(cfg, env)

# Train the model and save checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=cfg.algorithm_config.checkpoint_config.save_freq,
    save_path=os.path.join(cfg.artefact_path, "models/"),
    name_prefix=cfg.algorithm_config.experiment_name,
    save_replay_buffer=cfg.algorithm_config.checkpoint_config.save_replay_buffer,
    save_vecnormalize=cfg.algorithm_config.checkpoint_config.save_vecnormalize,
)

model.learn(
    total_timesteps=cfg.algorithm_config.learning_config.total_timesteps, 
    log_interval=cfg.algorithm_config.learning_config.log_interval, 
    tb_log_name=cfg.algorithm_config.experiment_name, 
    callback=checkpoint_callback
)