# Algorithm imports
from stable_baselines3 import SAC

def create_model(cfg, env):

    # Prepare model
    model = SAC(
        "MlpPolicy", 
        env,
        verbose=1, 
        device=cfg.algorithm_config.device, 
        seed=cfg.algorithm_config.seed,
        tensorboard_log=cfg.algorithm_config.tensorboard_config.tensorboard_base_dir,
        buffer_size=cfg.algorithm_config.max_buffer_size,
    )

    # Load checkpoints
    checkpoint_load_path = cfg.algorithm_config.checkpoint_load_path
    if checkpoint_load_path:
        model = SAC.load(checkpoint_load_path, env=env, device=cfg.algorithm_config.device)

    # Load replay buffer
    replay_buffer_load_path = cfg.algorithm_config.replay_buffer_load_path
    if replay_buffer_load_path:
        model.load_replay_buffer(replay_buffer_load_path)


    return model