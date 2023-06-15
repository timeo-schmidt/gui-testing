import time
def get_action(obs, env):
    time.sleep(0.2)
    return env.action_space.sample()