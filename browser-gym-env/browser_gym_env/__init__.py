from gymnasium.envs.registration import register

register(
    id="browser_gym_env/WebBrowserEnv-v0",
    entry_point="browser_gym_env.envs:WebBrowserEnv",
    nondeterministic=True
)