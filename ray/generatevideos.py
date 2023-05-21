import argparse

import gymnasium as gym
from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import Policy


from browser_gym_env.envs.web_browser_env import WebBrowserEnv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str,
        help="The path to the ray checkpoint folder")
    parser.add_argument("--num-videos", type=int, default=1,
        help="The number of videos that should be generated")
    parser.add_argument("--num-steps", type=int, default=20,
        help="The number of steps per video that should be generated")
    args = parser.parse_args()
    return args


def generate_videos_from_checkpoint(args):
    policy = Algorithm.from_checkpoint(args.checkpoint_path)
    env = gym.make('browser_gym_env/WebBrowserEnv-v0')

    for i in range(args.num_videos):
        print("Generating video: ", i)
        env.web_app_interface.start_recording()
        for j in range(args.num_steps):
            if j==0:
                obs, _ = env.reset()
            action = policy.compute_single_action(obs)
            print(action)
            obs, _, done, _, _ = env.step(action)
            if done:
                break
        env.web_app_interface.stop_recording()



if __name__ == "__main__":

    # Parse the command line arguments
    args = parse_args()

    # Generate videos from the checkpoint
    generate_videos_from_checkpoint(args)