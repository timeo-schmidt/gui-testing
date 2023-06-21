# Python standard libraries
import math
import random
import string
import time

# Third-party libraries
from PIL import Image, ImageDraw
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch as th

# Local application/library specific imports
from web_app_interface.web_app_interface import WebAppInterface

class WebBrowserEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, cfg, render_mode=None):
        # Ray-specific configuration
        if cfg.__class__.__name__ == "EnvContext":
            cfg = cfg["cfg"]

        # Configuration and settings
        self.cfg = cfg
        self.render_mode = render_mode
        self.viewport_size = cfg.web_app_interface.viewport_dimensions
        self.browser_reset_interval = cfg.web_app_interface.browser_reset_interval
        self.starting_url = cfg.env_config.test_url
        self.downscale_size = cfg.env_config.downscale_size

        # Environment logging and episode controls
        self.log_steps = cfg.env_config.log_steps
        self.episode_step_count = 0
        self.max_episode = cfg.env_config.horizon_length

        # Masking and grayscale settings
        self.masking = cfg.env_config.masking
        self.mask_centerpoint_only = cfg.env_config.mask_centerpoint_only
        self.masked_action_space_size = cfg.env_config.masked_action_space_size
        self.grayscale = cfg.env_config.grayscale
        n_img_channels = 1 if self.grayscale else 3

        # Define the observation space
        self._define_observation_space(n_img_channels)

        # Define the action space
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Initialize the WebAppInterface
        self.web_app_interface = None
        self._init_browser()

        # Video recording settings
        self.record_video = cfg.web_app_interface.record_video or (cfg.inference.record_video and cfg.mode=="test")
        if self.record_video:
            self.web_app_interface.start_recording()

        # Test system browser console logging flag
        self.log_errors = cfg.inference.log_errors

        # Interface properties
        self.inner_window_size = self.web_app_interface.browser.execute_script("return { width: window.innerWidth, height: window.innerHeight }")

        # Initialize reward
        self._init_reward()

        # Debugging properties
        self.env_id = ''.join(random.choice(string.ascii_lowercase) for i in range(5))
        
        self.prev_obs = None
        self.prev_elements = None

    def _define_observation_space(self, n_img_channels):
        if self.masking:
            self.observation_space = spaces.Dict({
                "screenshot": spaces.Box(low=0, high=255, shape=(self.downscale_size[1], self.downscale_size[0], n_img_channels), dtype=np.uint8),
                "clickable_elements": spaces.Box(low=0, high=255, shape=(self.masked_action_space_size[1], self.masked_action_space_size[0], 1), dtype=np.uint8)
            })
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.downscale_size[1], self.downscale_size[0], n_img_channels), dtype=np.uint8)


    """
    This function is called at the start and periodically to re-launch the browsers and free up accumulated memory.
    """
    def _init_browser(self):
        # Close the browser if it is already open
        if self.web_app_interface is not None:
            self.web_app_interface.browser.close()
        time.sleep(3)
        self.web_app_interface = WebAppInterface(self.cfg)#, starting_url=self.starting_url, detached=False, verbose=False)
        self.browser_open_steps = 0
        time.sleep(10)

    """
    This function gets a dict with the topLeft and bottomRight coordinates of the interactable elements in the current page.
    """
    def get_interactable_regions_dict(self):
        print("Getting interactable regions dict (masking)")
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
    
    def get_interactable_and_visible_xpaths(self):
        print("Getting interactable and visible xpaths...")
        expression = """
        (function() {
            function getXPathForElement(element) {
                const idx = (sib, name) => sib
                    ? idx(sib.previousElementSibling, name || sib.localName) + (sib.localName == name)
                    : 1;
                const segs = elm => !elm || elm.nodeType !== 1 
                    ? ['']
                    : elm.id && document.getElementById(elm.id) === elm
                        ? [`//*[@id="${elm.id}"]`]
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

            function isInteractable(element) {
                var interactableSelectors = ['a', 'button', 'input[type=button]', 'input[type=submit]'];
                var hasListeners = getEventListeners(element).click && getEventListeners(element).click.length > 0;
                var matchesSelector = interactableSelectors.some(selector => element.matches(selector));
                return hasListeners || matchesSelector;
            }

            function getVisibleAndInteractable() {
                const elements = document.querySelectorAll('*');
                const visibleInteractableElementsXPaths = [];
                for (let element of elements) {
                    if (isVisible(element) && isInteractable(element)) {
                        var xpath = getXPathForElement(element);
                        if ($x(xpath).length >= 1) {
                            visibleInteractableElementsXPaths.push(xpath);
                        }
                    }
                }
                return visibleInteractableElementsXPaths;
            }
        
            return getVisibleAndInteractable();
        })()
        """

        args = {
            "allowUnsafeEvalBlockedByCSP": False,
            "awaitPromise": False,
            "expression": expression,
            "generatePreview": False,
            "includeCommandLineAPI": True,
            "objectGroup": "console",
            "replMode": True,
            "returnByValue": True,
            "silent": False,
            "userGesture": True
        }

        result = self.web_app_interface.browser.execute_cdp_cmd("Runtime.evaluate", args)

        # Result from Javascript execution is in 'result' variable, under the 'result' key and then under the 'value' key
        interactable_elements = result['result']['value']

        return interactable_elements

    """
    This function retrieves an (image) mask with all the interactable elements in the current page.
    """
    def get_interactable_element_mask(self):
        # Get the interactable elements
        interactable_elements = self.get_interactable_regions_dict()

        # Initialize the mask tensor with zeroes
        tensor = np.zeros((self.masked_action_space_size[1], self.masked_action_space_size[0], 1), dtype=np.uint8)

        # Iterate over interactable elements and produce the mask
        for element in interactable_elements:

            # Skip the element if it covers more than 30% of the viewport
            if (element['bottomRight']['x'] - element['topLeft']['x']) * (element['bottomRight']['y'] - element['topLeft']['y']) > 0.5 * self.viewport_size[0] * self.viewport_size[1]:
                continue

            topLeft = element['topLeft']
            bottomRight = element['bottomRight']

            # Make sure the coordinates are within the viewport
            topLeft['x'] = max(0, min(self.viewport_size[0] - 1, topLeft['x']))
            topLeft['y'] = max(0, min(self.viewport_size[1] - 1, topLeft['y']))
            bottomRight['x'] = max(0, min(self.viewport_size[0] - 1, bottomRight['x']))
            bottomRight['y'] = max(0, min(self.viewport_size[1] - 1, bottomRight['y']))

            # Scale the coordinates to the inner window size
            topLeft['x'] = math.ceil(topLeft['x'] * self.masked_action_space_size[0] / self.inner_window_size['width'])
            topLeft['y'] = math.ceil(topLeft['y'] * self.masked_action_space_size[1] / self.inner_window_size['height'])
            bottomRight['x'] = math.floor(bottomRight['x'] * self.masked_action_space_size[0] / self.inner_window_size['width'])
            bottomRight['y'] = math.floor(bottomRight['y'] * self.masked_action_space_size[1] / self.inner_window_size['height'])

            if self.mask_centerpoint_only:
                # Find the center points
                center = {
                    'x': int((topLeft['x'] + bottomRight['x']) / 2),
                    'y': int((topLeft['y'] + bottomRight['y']) / 2)
                }
                
                # Clip the coordinates to the viewport size
                center['x'] = max(0, min(self.masked_action_space_size[0] - 1, center['x']))
                center['y'] = max(0, min(self.masked_action_space_size[1] - 1, center['y']))

                tensor[center['y'], center['x'], 0] = 255
            else:
                # Update the tensor 
                tensor[topLeft['y']:bottomRight['y'], topLeft['x']:bottomRight['x']] = 255
                
        return tensor

    """
    Get an observation of the current state of the environment.
    """
    def _get_obs(self):

        # Get a screenshot of the browser window and downscale it to the downscale_size
        screenshot = self.web_app_interface.get_screenshot(size=self.downscale_size)

        # Convert to a numpy array and remove the alpha/transparency channel
        screenshot = np.asarray(screenshot)[...,:3]

        # Convert the screenshot to grayscale using cv2
        if self.grayscale:
            screenshot = np.dot(screenshot[...,:3], [0.2989, 0.5870, 0.1140])
            screenshot = screenshot.astype(np.uint8)
            screenshot = np.expand_dims(screenshot, axis=-1)

        ############### DEBUG ###############
        # Get the screenshot and mask of the previous observation
        # Make sure the data types are uint8
        # ss = screenshot.astype(np.uint8)        
        # mask = mask.astype(np.uint8)
        # # Convert arrays to Pillow Images
        # ss = ss.squeeze()
        # ms = mask.squeeze()
        # screenshot_img = Image.fromarray(ss, 'RGB')  # assuming screenshot is in RGB format
        # mask_img = Image.fromarray(ms, 'L')  # assuming mask is grayscale
        # # Resize the screenshot to the mask size
        # screenshot_img = screenshot_img.resize(mask_img.size)
        # # Now apply the mask to the screenshot
        # screenshot_img.putalpha(mask_img)
        # # Save the image with env id and random number
        # screenshot_img.save(f"{self.env_id}_.png")
        ############### END DEBUG ###############

        if self.masking:
            mask = self.get_interactable_element_mask()
            obs = { 
                "screenshot": screenshot,
                "clickable_elements": mask
            }
        else:
            obs = screenshot

        self.prev_obs = obs
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
    def _calculate_reward_legacy(self, is_deadend=False):

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
    The reward function for the environment.
    """
    def _calculate_reward_previously_unseen(self):

        all_visible_xpaths = set(self._get_visible_paths())

        # Get the new xpaths that have not been seen previously in self.known_xpaths
        new_xpaths = all_visible_xpaths - self.known_xpaths

        # Add the new xpaths to the known xpaths
        self.known_xpaths = self.known_xpaths.union(new_xpaths)

        reward = len(new_xpaths)
        
        return reward
    
    """
    Reward function that calculates reward based on the visual difference between the previous and current observation.
    """
    def _calculate_visual_reward(self, prev_obs, obs):
        # Calculate the difference between the frames
        diff = np.abs(prev_obs - obs)
        
        # Calculate the sum of the difference
        reward = np.sum(diff)

        return reward
    
    """
    Reward function that calculates the reward based on the delta of the number of visible elements.
    """
    def _calculate_element_delta_reward(self):

        # Get the current visible xpaths
        current_elements = set(self._get_visible_paths())

        if self.prev_elements is None:
            # If this is the first time step, then set the known xpaths to the current visible xpaths
            self.prev_elements = current_elements
            return 0
        else:
            # Get the delta of visible elements
            delta = len(current_elements - self.prev_elements)

            # Update the previous elements
            self.prev_elements = current_elements

            return delta
        
    def apply_reward_addons(self, reward, truncated):
        # Termination penalty
        if truncated:
            return -10 # Return a large negative reward

        # Logarithmic scaling
        if self.cfg.env_config.reward_addon_logarithmic_scaling:
            reward = np.log(reward+1.0)

        # Scale Factor
        reward *= self.cfg.env_config.reward_addon_scale_factor

        # Negative default
        if self.cfg.env_config.reward_addon_negative_default:
            if reward == 0:
                reward = self.cfg.env_config.reward_addon_negative_default

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
        try:
            self.web_app_interface.browser.execute_script('throw Error("This is a test error");')
        except:
            pass

        # Convert the action space representation to pixel click locations
        x = int((action[0]*0.5+0.5)*self.viewport_size[0])
        y = int((action[1]*0.5+0.5)*self.viewport_size[1])

        # Perform the mouse click
        self.web_app_interface.click(x, y)

        # Calcualte a new observation and info
        prev_obs = self.prev_obs
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


        # Calculate the reward
        reward_variant = self.cfg.env_config.reward_variant
        if reward_variant == 1:
            # Variant 1: Reward based on visual difference (pixel difference)
            reward = self._calculate_visual_reward(prev_obs, observation)
        elif reward_variant == 2:
            # Variant 2: Reward based on the delta of the number of visible elements
            reward = self._calculate_element_delta_reward()
        elif reward_variant == 3:
            # Variant 3: Reward based on the number of new, previously unseen elements
            reward = self._calculate_reward_previously_unseen()

        # Apply addons to the reward
        reward = self.apply_reward_addons(reward, is_deadend)

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
            x_scaled = int(x * mask.shape[0] / self.viewport_size[0])
            y_scaled = int(y * mask.shape[1] / self.viewport_size[1])

            draw.ellipse((y_scaled-5, x_scaled-5, y_scaled+5, x_scaled+5), fill='red', outline='red')

            # Save the image with the env id as name
            screenshot_img.save(f"env_{self.env_id}.png")

        if self.log_steps:
            # Print episode, action, reward all fixed length
            print(f"Episode: {self.episode_step_count:03d} | Action: ({x:04d}, {y:04d}) | Reward: {reward:03f}")

        # Increment browser open steps
        self.browser_open_steps += 1
        if(self.browser_open_steps >= self.browser_reset_interval):
            self._init_browser()

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
        if self.log_errors:
            self.web_app_interface.write_log_file()
        self.web_app_interface.browser.quit()