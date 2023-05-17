from gymnasium.envs.registration import register

register(
    id="browser_gym_env/WebBrowserEnv-v0",
    entry_point="browser_gym_env.envs:WebBrowserEnv",
    nondeterministic=True
)

def test_web_app_interface_only():
    from .envs.web_app_interface.web_app_interface import WebAppInterface
    wai = WebAppInterface()
