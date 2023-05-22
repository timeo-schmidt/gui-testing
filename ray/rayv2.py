import ray
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.a2c.a2c import A2C
from ray.rllib.algorithms.a2c import A2CConfig
from ray.rllib.utils.annotations import override
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided

from browser_gym_env.envs.web_browser_env import WebBrowserEnv

ray.init()

"""
Function that retrieves the custom policy class
"""
def custom_get_custom_policy_class():
    print("custom policy class is in use!")
    from ray.rllib.algorithms.a3c.a3c_tf_policy import A3CTF1Policy
    return A3CTF1Policy

"""
Function that retrieves the custom config
"""
def custom_get_custom_config():
    print("custom config is in use!")
    from ray.rllib.algorithms.a2c import A2CConfig
    config = (
        A2CConfig()
        .training(replay_buffer_config={"capacity": 5000})
        .framework("tf")
        .environment(WebBrowserEnv, observation_space=spaces.Box(low=0, high=1, shape=(240, 320, 3), dtype=np.float32))
        .rollouts(num_rollout_workers=0, num_envs_per_worker=1)
        .resources(num_gpus=0)
        .offline_data(output="./experience_recordings/")
    )
    return config


# Create a new Algorithm using the Policy defined above.
class CustomA2C(A2C):
    @classmethod
    def default_config(cls):
        return custom_get_custom_config()

    @classmethod
    def get_custom_policy_class(cls, config):
        return custom_get_custom_policy_class()


algo = CustomA2C()


for i in range(5000):
    print("TRAINING CYCLE: ", i)
    result = algo.train()
    print(pretty_print(result))

    if i % 20 == 0:
        checkpoint = algo.save()
        print("checkpoint saved at", checkpoint)

algo.stop()