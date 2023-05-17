import numpy as np
import gymnasium as gym
from gymnasium import spaces

from browser_gym_env.envs.web_app_interface.web_app_interface import WebAppInterface

# Environment Settings
VIEWPORT_SIZE = (1080, 720)                 # The size of the browser window in pixels as a tuple: (width, height)
DOWNSCALE_SIZE = (320, 240)                 # The size of the downsampled screenshot in pixels as a tuple: (width, height)
STARTING_URL = "http://localhost:3000/"     # The URL to load when the environment is reset
MAX_EPISODE = 20                            # The maximum number of steps in an episode

class WebBrowserEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        self.viewport_size = VIEWPORT_SIZE
        self.downscale_size = DOWNSCALE_SIZE
        self.starting_url = STARTING_URL

        # The observation space is the downsampled screenshot of the browser window in RGB
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.downscale_size[1], self.downscale_size[0], 3), dtype=np.float32)

        # The action space is continuous and consists of two numbers: the x and y click coordinates
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

        # assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Store the current episode step count
        self.episode_step_count = 0
        self.max_episode = MAX_EPISODE

        # Initialize the WebAppInterface
        self.web_app_interface = WebAppInterface(screen_size=self.viewport_size, starting_url=self.starting_url, detached=False)

        # Store the elements that are visible in the starting state
        self.unique_visible_elements = set(self.web_app_interface.get_all_elements())

    def _get_obs(self):

        # Get a screenshot of the browser window and downscale it to the downscale_size
        obs = self.web_app_interface.get_screenshot(size=self.downscale_size)

        # Convert to a numpy array and remove the alpha/transparency channel
        obs = np.asarray(obs)[...,:3]

        # normalise by dividing by 255
        obs = obs / 255.0

        return obs

    def _get_info(self):
        return {
            "url": self.web_app_interface.browser.current_url,
            "elements": self.web_app_interface.get_all_elements()
        }

    def reset(self, *, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Navigate the browser to the starting URL
        self.web_app_interface.browser.get(self.starting_url)

        # Get the starting state screenshot and auxillary info
        observation = self._get_obs()
        info = self._get_info()

        # Reset the unique visible elements
        self.unique_visible_elements = set(self.web_app_interface.get_all_elements())

        # if self.render_mode == "human":
        #    self._render_frame()

        return observation, info

    def filter_visible_web_elements(self, web_element_list):
        visible_elements = []
        for element in web_element_list:
            try:
                if element.is_displayed():
                    visible_elements.append(element)
            except:
                pass
        return set(visible_elements)


    """
    The reward function is equal to the newly discovered visible elements count
    """
    def _calculate_reward(self, is_deadend=False):
        # Get all the new elements that have not been seen previously in self.unique_visible_elements
        new_elements = set(self.web_app_interface.get_all_elements()) - self.unique_visible_elements
        # Get the visible elements from the new elements
        visible_new_elements = self.filter_visible_web_elements(new_elements)
        # Add the visible new elements to the set of unique visible elements
        self.unique_visible_elements = self.unique_visible_elements.union(visible_new_elements)

        # The reward is the number of new elements that have been seen
        reward = len(visible_new_elements)

        if (reward == 0):
            reward = -0.01
        elif is_deadend:
            reward = -1
        else:
            reward = np.log(reward+1)*10
        
        return reward

    def step(self, action):
        # Convert the action space representation to pixel click locations
        x = int(action[0]*self.viewport_size[0])
        y = int(action[1]*self.viewport_size[1])
        # Click the mouse at the specified location
        self.web_app_interface.click(x, y)

        # An episode is done if the max episode step count is reached

        is_deadend = bool(self.web_app_interface.fix_deadends())

        terminated = self.episode_step_count >= self.max_episode or is_deadend
        if not terminated:
            self.episode_step_count += 1
        else:
            self.episode_step_count = 0
            # Save a screenshot
            self.web_app_interface.save_screenshot("ss.png")
        
        truncated = terminated

        # The reward is equal to the number of newly discovered elements
        reward = self._calculate_reward(is_deadend)
        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        # Print episode, action, reward all fixed length
        print(f"Episode: {self.episode_step_count:03d} | Action: ({x:04d}, {y:04d}) | Reward: {reward:03f}")

        return observation, reward, terminated, truncated, info

    def render(self):
        # The rendering is not done separately, it is always done in the browser window
        return self._get_obs()

    def close(self):
        self.web_app_interface.browser.quit()
