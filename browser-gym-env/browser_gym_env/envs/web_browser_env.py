import time
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces

import random
import string

from PIL import Image, ImageDraw

from browser_gym_env.envs.web_app_interface.web_app_interface import WebAppInterface

# Environment Settings
VIEWPORT_SIZE = (1080, 720)                 # The size of the browser window in pixels as a tuple: (width, height)
DOWNSCALE_SIZE = (300, 200)                 # The size of the downsampled screenshot in pixels as a tuple: (width, height)
STARTING_URL = "http://localhost:3000/"     # The URL to load when the environment is reset
MAX_EPISODE = 20                            # The maximum number of steps in an episode (aka Horizon)
BROWSER_RESET_INTERVAL = 901                # The number of steps after which the browser is re-launched to free up memory (using non random number to make error attribution easier)
DISCRETISED_ACTION_SPACE_SIZE = (300, 200)

class WebBrowserEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, masking=False, log_steps=True, reward_clickable=True, record_video=False):
        self.viewport_size = VIEWPORT_SIZE
        self.downscale_size = DOWNSCALE_SIZE
        self.starting_url = STARTING_URL
        
        self.masking = masking
        self.reward_clickable = reward_clickable
        self.log_steps = log_steps

        # Depending on the masking flag, the observation space is either only the screenshot or a dictionary of the screenshot and mask
        if self.masking:
            self.observation_space = spaces.Dict({
                "screenshot": spaces.Box(low=0, high=255, shape=(self.downscale_size[1], self.downscale_size[0], 3), dtype=np.uint8),
                "clickable_elements": spaces.Box(low=0, high=255, shape=(DISCRETISED_ACTION_SPACE_SIZE[1], DISCRETISED_ACTION_SPACE_SIZE[0], 1), dtype=np.uint8)
            })
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.downscale_size[1], self.downscale_size[0], 3), dtype=np.uint8)

        # The action space is continuous and consists of two numbers: the x and y click coordinates
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.render_mode = render_mode

        # Store the current episode step count
        self.episode_step_count = 0
        self.max_episode = MAX_EPISODE

        # Initialize the WebAppInterface
        self.web_app_interface = None
        self._init_browser()

        # Start the video recording if the flag is set
        self.record_video = record_video
        if self.record_video:
            self.web_app_interface.start_recording()

        # Calculate and store the inner window size through JS call
        self.inner_window_size = self.web_app_interface.browser.execute_script("return { width: window.innerWidth, height: window.innerHeight }")

        # Reset reward
        self._init_reward()

        # Following lines are for debugging purposes
        # Generate a random 5-letter hash using python
        self.env_id = ''.join(random.choice(string.ascii_lowercase) for i in range(5))
        self.prev_obs = None

    """
    This function is called at the start and periodically to re-launch the browsers and free up accumulated memory.
    """
    def _init_browser(self):
        # Close the browser if it is already open
        if self.web_app_interface is not None:
            self.web_app_interface.browser.close()
        time.sleep(3)
        self.web_app_interface = WebAppInterface(screen_size=self.viewport_size, starting_url=self.starting_url, detached=False, verbose=False)
        self.browser_open_steps = 0
        time.sleep(10)

    """
    This function gets a dict with the topLeft and bottomRight coordinates of the interactable elements in the current page.
    """
    def get_interactable_regions_dict(self):
        expression = """
        function getElementsPositions() {
            var allElements = document.querySelectorAll('*');
            var interactableSelectors = ['a', 'button', 'input[type=button]', 'input[type=submit]'];
            var interactableElements = [];

            allElements.forEach(element => {
                var listeners = getEventListeners(element);
                if (listeners.click && listeners.click.length > 0) {
                interactableElements.push(element);
                }
            });

            interactableSelectors.forEach(selector => {
                var elements = document.querySelectorAll(selector);
                elements.forEach(element => {
                if (!interactableElements.includes(element)) {
                    interactableElements.push(element);
                }
                });
            });

            var elementPositions = [];
            interactableElements.forEach(element => {
                var rect = element.getBoundingClientRect();
                var position = {
                topLeft: { x: parseInt(rect.left), y: parseInt(rect.top) },
                bottomRight: { x: parseInt(rect.right), y: parseInt(rect.bottom) }
                };
                elementPositions.push(position);
            });

            return elementPositions;
        }
        var interactable_elements = getElementsPositions();
        """

        args = {
            "allowUnsafeEvalBlockedByCSP": False,
            "awaitPromise": False,
            "expression": expression,
            "generatePreview": True,
            "includeCommandLineAPI": True,
            "objectGroup": "console",
            "replMode": True,
            "returnByValue": False,
            "silent": False,
            "userGesture": True
        }

        # Get all the clickable elements
        self.web_app_interface.browser.execute_cdp_cmd("Runtime.evaluate", args)

        # Now use normal js to return the clickable elements
        interactable_elements = self.web_app_interface.browser.execute_script("return interactable_elements;")

        return interactable_elements
        

    """
    This function retrieves an (image) mask with all the interactable elements in the current page.
    """
    def get_interactable_element_mask(self):

        # Get the interactable elements
        interactable_elements = self.get_interactable_regions_dict()

        # Initialize the mask tensor with zeroes
        tensor = np.zeros((VIEWPORT_SIZE[1], VIEWPORT_SIZE[0], 1), dtype=np.uint8)

        # Iterate over interactable elements and produce the mask
        for element in interactable_elements:

            # Skip the element if it covers more than 30% of the viewport
            if (element['bottomRight']['x'] - element['topLeft']['x']) * (element['bottomRight']['y'] - element['topLeft']['y']) > 0.5 * VIEWPORT_SIZE[0] * VIEWPORT_SIZE[1]:
                continue

            topLeft = element['topLeft']
            bottomRight = element['bottomRight']

            # Make sure the coordinates are within the viewport
            topLeft['x'] = max(0, min(VIEWPORT_SIZE[0] - 1, topLeft['x']))
            topLeft['y'] = max(0, min(VIEWPORT_SIZE[1] - 1, topLeft['y']))
            bottomRight['x'] = max(0, min(VIEWPORT_SIZE[0] - 1, bottomRight['x']))
            bottomRight['y'] = max(0, min(VIEWPORT_SIZE[1] - 1, bottomRight['y']))

            # Scale the coordinates to the inner window size
            topLeft['x'] = int(topLeft['x'] * VIEWPORT_SIZE[0] / self.inner_window_size['width'])
            topLeft['y'] = int(topLeft['y'] * VIEWPORT_SIZE[1] / self.inner_window_size['height'])
            bottomRight['x'] = int(bottomRight['x'] * VIEWPORT_SIZE[0] / self.inner_window_size['width'])
            bottomRight['y'] = int(bottomRight['y'] * VIEWPORT_SIZE[1] / self.inner_window_size['height'])

            # Update the tensor 
            tensor[topLeft['y']:bottomRight['y'], topLeft['x']:bottomRight['x']] = 255
            
        if VIEWPORT_SIZE != DISCRETISED_ACTION_SPACE_SIZE:
            # Resize the tensor to the downscale size with pillow
            tensor = tensor.squeeze().astype(np.uint8)
            tensor = Image.fromarray(tensor, mode='L')
            tensor = tensor.resize(DISCRETISED_ACTION_SPACE_SIZE, resample=Image.BILINEAR)
            tensor = np.asarray(tensor)
            tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], 1)

        # Set all the values that are not equal to 255 to 0
        tensor = np.where(tensor == 255, tensor, 0)

        return tensor

    """
    Get an observation of the current state of the environment.
    """
    def _get_obs(self):

        # Get a screenshot of the browser window and downscale it to the downscale_size
        screenshot = self.web_app_interface.get_screenshot(size=self.downscale_size)

        # Convert to a numpy array and remove the alpha/transparency channel
        screenshot = np.asarray(screenshot)[...,:3]

        # if self.masking:
        #     mask = self.get_interactable_regions_dict()

        # # Normalize the values to be between 0 and 1
        # obs = obs / 255.0

        # # Remove any non-finite values
        # obs = np.nan_to_num(obs, copy=False, nan=0.0, posinf=1.0, neginf=0)

        # # as a dummy, create a random clickable elements array with dtyp uint8
        # # Create an array where the top half is 0 and the bottom half is 255 with dtype uint8
        # clickable_elements = np.zeros((VIEWPORT_SIZE[1], VIEWPORT_SIZE[0], 1), dtype=np.uint8)
        # clickable_elements[700:,700:] = 255

        # # Get the screenshot and mask of the previous observation
        # # Make sure the data types are uint8
        # ss = screenshot.astype(np.uint8)
        # mask = mask.astype(np.uint8)

        # # Convert arrays to Pillow Images
        # screenshot_img = Image.fromarray(ss, 'RGB')  # assuming screenshot is in RGB format
        # mask_img = Image.fromarray(mask.squeeze(), 'L')  # assuming mask is grayscale

        # # Resize the screenshot to the mask size
        # screenshot_img = screenshot_img.resize(mask_img.size)

        # # Now apply the mask to the screenshot
        # screenshot_img.putalpha(mask_img)

        # # Save the image with env id and random number
        # screenshot_img.save(f"{self.env_id}_.png")

        if self.masking:
            obs = { 
                "screenshot": screenshot,
                "clickable_elements": self.get_interactable_element_mask()
            }
        else:
            obs = screenshot

        return obs

    """
    Not used, hence just returns an empty dict.
    """
    def _get_info(self):
        return dict()

    """
    This function resets the state of the reward calculation.
    """
    def _init_reward(self):
        self.known_xpaths = set(self._get_visible_paths())

    """
    Resets the environemtn
    """
    def reset(self, seed=None, options=None):
        # Navigate the browser to the starting URL
        self.web_app_interface.browser.get(self.starting_url)

        # Get the starting state screenshot and auxillary info
        observation = self._get_obs()
        info = self._get_info()

        # Reset reward
        self._init_reward()

        return observation, info
    
    """
    This is a helper for the reward calculation and it computes all the visible xpaths on the current page.
    """
    def _get_visible_paths(self):
        xpaths = self.web_app_interface.browser.execute_script("""
        function getXPathForElement(element) {
            const idx = (sib, name) => sib
                ? idx(sib.previousElementSibling, name || sib.localName) + (sib.localName == name)
                : 1;
            const segs = elm => !elm || elm.nodeType !== 1 
                ? ['']
                : elm.id && document.getElementById(elm.id) === elm
                    ? [`id("${elm.id}")`]
                    : [...segs(elm.parentNode), `${elm.localName.toLowerCase()}`];
            let path = segs(element);
            path[path.length - 1] = `${path[path.length - 1]}[${idx(element)}]`;
            return path.join('/');
        }

        function isVisible(element) {
            const style = window.getComputedStyle(element);
            const rect = element.getBoundingClientRect();

            return style.display !== 'none' &&
                style.visibility !== 'hidden' &&
                parseFloat(style.opacity) > 0 &&
                rect.width > 0 && rect.height > 0 &&
                rect.top >= 0 && rect.left >= 0 &&
                (rect.bottom + window.scrollY) <= (window.innerHeight + window.scrollY) &&
                (rect.right + window.scrollX) <= (window.innerWidth + window.scrollX);
        }

        const elements = document.querySelectorAll('*');
        const visibleElementsXPaths = [];

        for (let element of elements) {
            if (isVisible(element)) {
                visibleElementsXPaths.push(getXPathForElement(element));
            }
        }
        return visibleElementsXPaths;
        """)

        return xpaths

    """
    The reward function for the environment.
    """
    def _calculate_reward(self, actions, is_deadend=False):

        all_visible_xpaths = set(self._get_visible_paths())

        # Get the new xpaths that have not been seen previously in self.known_xpaths
        new_xpaths = all_visible_xpaths - self.known_xpaths

        # Add the new xpaths to the known xpaths
        self.known_xpaths = self.known_xpaths.union(new_xpaths)

        reward = len(new_xpaths)

        if (reward == 0):
            reward = -0.01
        elif is_deadend:
            reward = -10
        else:
            reward = np.log(reward+1.0)*10.0
        
        return reward
    
    """
    Check that the observation is not just a white image or an image with no clickable elements.
    """
    def _check_obs_valid(self, obs):
        if self.masking:
            return not np.all(obs["screenshot"] == 255) and not np.all(obs["clickable_elements"] == 0)
        else:
            return not np.all(obs == 255)
       

    """
    One time step in the environment.
    """
    def step(self, action):

        # Convert the action space representation to pixel click locations
        x = int((action[0]*0.5+0.5)*self.viewport_size[0])
        y = int((action[1]*0.5+0.5)*self.viewport_size[1])

        # Perform the mouse click
        self.web_app_interface.click(x, y)

        # Calcualte a new observation and info
        observation = self._get_obs()
        info = self._get_info()

        # Check wether the observation is valid
        is_obs_valid = self._check_obs_valid(observation)

        if not is_obs_valid:
            print("Invalid observation")

        # After every click, check that the page has not become a deadend
        is_deadend = bool(self.web_app_interface.fix_deadends())

        # If the page is a deadend, then truncate the episode
        truncated = is_deadend or not is_obs_valid

        # An episode is done if the max episode step count is reached or the page is a deadend
        terminated = self.episode_step_count >= self.max_episode or is_deadend

        # Compute the reward
        reward = self._calculate_reward([x,y], is_deadend)

        if not terminated:
            self.episode_step_count += 1
        else:
            self.episode_step_count = 0


        if self.prev_obs is not None and self.masking and False:
            # Get the screenshot and mask of the previous observation
            screenshot = self.prev_obs["screenshot"]
            mask  = self.prev_obs["clickable_elements"]
            # print the shapes in a single line nicely
            # print(f"Shapes: {screenshot.shape}, {mask.shape}") # Shapes: (128, 128, 3), (720, 1080, 1)

            # Make sure the data types are uint8
            screenshot = screenshot.astype(np.uint8)
            mask = mask.astype(np.uint8)

            # Convert arrays to Pillow Images
            screenshot_img = Image.fromarray(screenshot, 'RGB')  # assuming screenshot is in RGB format
            mask_img = Image.fromarray(mask.squeeze(), 'L')  # assuming mask is grayscale

            # Resize the screenshot to the mask size
            screenshot_img = screenshot_img.resize(mask_img.size)

            # Now apply the mask to the screenshot
            screenshot_img.putalpha(mask_img)

            # Now draw a red circle at the click location
            draw = ImageDraw.Draw(screenshot_img)
            # Scale the click location to the mask size
            x_scaled = int(x * mask.shape[0] / VIEWPORT_SIZE[0])
            y_scaled = int(y * mask.shape[1] / VIEWPORT_SIZE[1])

            draw.ellipse((y_scaled-5, x_scaled-5, y_scaled+5, x_scaled+5), fill='red', outline='red')

            # Save the image with the env id as name
            screenshot_img.save(f"env_{self.env_id}.png")

        self.prev_obs = observation

        if self.log_steps:
            # Print episode, action, reward all fixed length
            print(f"Episode: {self.episode_step_count:03d} | Action: ({x:04d}, {y:04d}) | Reward: {reward:03f}")

        # Increment browser open steps
        self.browser_open_steps += 1
        if(self.browser_open_steps >= BROWSER_RESET_INTERVAL):
            self._init_browser()

        # Include a delay if a video is being recorded
        if self.record_video:
            time.sleep(1)

        return observation, reward, terminated, truncated, info

    """
    This is not used, but included as it is a standard gym environment method.
    """
    def render(self):
        pass
    
    """
    This function closes the browser.
    """
    def close(self):
        if self.record_video:
            self.web_app_interface.stop_recording()
        self.web_app_interface.browser.quit()
