# Algorithm imports
from stable_baselines3 import SAC

# Environment imports
import browser_gym_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

MODEL_PATH = "./models/sac_vanilla_500k_100000_steps.zip"

# Prepare environment
env = make_vec_env(
    "browser_gym_env/WebBrowserEnv-v0", 
    n_envs=1, 
    vec_env_cls=SubprocVecEnv, 
    env_kwargs={"masking": False, "log_steps": True, "record_video":True},
    vec_env_kwargs=dict(start_method='fork')
)

# env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)

# Prepare model
model = SAC(
    "CnnPolicy", 
    env,
    verbose=1, 
    device="mps", 
    seed=42,
    tensorboard_log="./tensorboard/",
)

# Load checkpoints
model = SAC.load(MODEL_PATH, env=env, device="mps")

# Evaluate the mode
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1, deterministic=True)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

env.close()