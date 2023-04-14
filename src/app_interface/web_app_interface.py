from .abstract import AbstractAppInterface
from overrides.preambles import inject_preamble
from overrides.deadends import recover_deadend
import random
import time
import os
from PIL import Image, ImageDraw
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.actions.action_builder import ActionBuilder

import numpy as np
import cv2
import threading

from mss import mss
from dataclasses import dataclass

class WebAppInterface(AbstractAppInterface):

    """
    This class is an interface for interacting with web apps
    Parameters:
        screen_size: The size of the browser window to open
        artifact_path: The path to save the screenshots to
        starting_url: The url to open when the browser is opened
        detached: Whether the browser should be detached from the python process (stays open after the script is finished)
        verbose: Whether to print out the actions being performed
    """

    def __init__(self, screen_size=(1920, 1080), artifact_path="./screenshots", starting_url="https://www.youtube.com/", detached=True, verbose=True):
        # Options passed to browser
        options = webdriver.ChromeOptions()
        options.add_argument("window-size=" + str(screen_size[0]) + "," + str(screen_size[1]))
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_experimental_option("detach", detached)
        prefs = {"credentials_enable_service": False,
            "profile.password_manager_enabled": False}
        options.add_experimental_option("prefs", prefs)
        
        # Initialise the selenium web driver
        self.browser = webdriver.Chrome(ChromeDriverManager().install(), options=options)
        # Call the base url
        self.browser.get(starting_url)
        self.starting_url = starting_url

        # Check if the artifact path exists, if not create it
        if not os.path.exists(artifact_path):
            os.makedirs(artifact_path)
        # Set the artifact path
        self.artifact_path = artifact_path

        self.action_history = []
        self.start_time = time.time()
        self.verbose = verbose

        # Get the browser window location
        self.window_location = self.browser.get_window_rect()

        # Based on the get_window_size, get the actual viewport location on the screen
        self.viewport_size = self.get_window_size()

        # Inject the preamble
        inject_preamble(self)

    """
    Adding the data classess for the different action types with their respective parameters
    """
    @dataclass
    class MouseMove:
        t: float
        x: int
        y: int
        def paint_interaction(self, image, alpha):
            return image # Mouse movements are not painted

    @dataclass
    class MouseClick:
        t: float
        x: int
        y: int
        def paint_interaction(self, image, alpha):
            overlay = image.copy()
            overlay = cv2.circle(overlay, (self.x, self.y), 7, (255,0,0), -1)
            return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    @dataclass
    class MouseSlide:
        t: float
        x1: int
        y1: int
        x2: int
        y2: int
        def paint_interaction(self, image, alpha):
            overlay = image.copy()
            overlay = cv2.line(overlay, (self.x1, self.y1), (self.x2, self.y2), (255,0,0), 7)
            return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    """
    This function returns the type of app that this interface is for as a string
    """
    def get_app_type(self):
        return "web"
    
    """
    This function clips the coordinates to the size of the window to ensure they stay within the window (0,size)
    """
    def _clip_coordinates(self, x, y):
        window_size = self.get_window_size()
        x = max(5, min(x, window_size["width"]-5))
        y = max(5, min(y, window_size["height"]-5))
        return x,y

    """
    This function returns the size of the app window as a dict e.g.: {'height': 421, 'width': 500}
    The window size is the size of the viewport, not the size of the browser window
    This means that the search bar and browser buttons are not included in the size
    """
    def get_window_size(self):
        # Get the viewport size using js
        viewport_size = self.browser.execute_script("return {width: window.innerWidth, height: window.innerHeight}")
        return viewport_size

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    """
    This function moves the mouse to (x,y)
    """ 
    def move_mouse(self, x, y):
        x,y = self._clip_coordinates(x, y)
        if self.verbose:
            self._print(f"Moving mouse to {x}, {y}")
        action_start_time = time.time()
        action = ActionBuilder(self.browser)
        action.pointer_action.move_to_location(x, y)
        action.perform()
        # only append to action history if called from outside
        self.action_history.append(self.MouseMove(action_start_time, x, y))

    """
    This function performs a mouse click at (x,y)
    """
    def click(self, x, y):
        x,y = self._clip_coordinates(x, y)
        self._print(x,y)
        if self.verbose:
            self._print(f"Clicking at {x}, {y}")
        self.move_mouse(x, y)
        action_start_time = time.time()
        ActionChains(self.browser).click().perform()
        self.action_history.append(self.MouseClick(action_start_time, x, y))

    """
    This function performs a mouse slide from (x,y) to (x2,y2)
    """
    def slide(self, x, y, x2, y2):
        x,y = self._clip_coordinates(x, y)
        x2,y2 = self._clip_coordinates(x2, y2)
        if self.verbose:
            self._print(f"Sliding from {x}, {y} to {x2}, {y2}")
        action_start_time = time.time()
        self.move_mouse(x, y)
        ActionChains(self.browser).click_and_hold().move_by_offset(x2-x, y2-y).release().perform()
        self.action_history.append(self.MouseSlide(action_start_time, x, y, x2, y2))

    """
    This function returns the current state of the browser window as a string, which contains the html code of the page
    """
    def get_gui_state(self):
        return self.browser.page_source
    
    """
    This function recovers from any known deadend states by manually injecting the actions to recover from the deadend
    """
    def fix_deadends(self):
        recover_deadend(self)

    """
    This function returns the screenshot of the browser window.
    It uses the mss library to capture the screenshot of the browser window efficiently.
    The screenshot is taken by using the window location and the viewport size to calculate the correct area to capture.
    Therefore it is important not to move or resize the browser window during the screen recording.
    """
    def get_screenshot(self, size=None):
        with mss() as sct:
            monitor = {
                "top": self.window_location["y"] + (self.window_location["height"] - self.viewport_size["height"]),
                "left": self.window_location["x"] + (self.window_location["width"] - self.viewport_size["width"]),
                "width": self.viewport_size["width"],
                "height": self.viewport_size["height"],
            }

            # Capture the screenshot
            sct_img = sct.grab(monitor)
            # Convert it to an Image object using Image.frombuffer()
            image = Image.frombuffer("RGB", sct_img.size, sct_img.rgb, "raw", "RGB", 0, 1)

            # Resize the image if a size is specified
            if size is not None:
                image = image.resize(size)
            
            return image
    

    """
    This function saves the screenshot to the artifact path
    """
    def save_screenshot(self, filename=time.time()):
        save_path = os.path.join(self.artifact_path, str(filename))
        self.browser.save_screenshot(save_path)

    """
    This function returns the action history
    """
    def get_action_history(self):
        return self.action_history

    """
    This helper function returns the time since the start of the recording
    """
    def _get_frame(self):
        frame = self.get_screenshot()
        frame_np = np.array(frame)
        frame_np = frame_np[:, :, [2, 1, 0]]
        return frame_np

    """
    This function starts the recording of the screen (in a separate thread)
    """
    def start_recording(self, filename=time.time()):
        vid_path = os.path.join(self.artifact_path, "screen_recording/", str(filename) + ".mp4")
        if not os.path.exists(os.path.dirname(vid_path)):
            os.makedirs(os.path.dirname(vid_path))
        self.recording_filename = vid_path

        # Start the thread
        self.stop_recording_flag = False
        self.recording_thread = threading.Thread(target=self._record_screen_thread)
        self.recording_thread.start()

    """
    This function is the thread that records the screen
    """
    def _record_screen_thread(self):
        self.frames = []
        target_frame_duration = 1 / 30  # Target frame duration for 30 FPS
        while not self.stop_recording_flag:
            frame_start_time = time.time()  # Record the time when the frame capture starts
            frame = self._get_frame()
            frame_duration = time.time() - frame_start_time  # Calculate the time it took to capture the frame
            self.frames.append((frame, frame_start_time))

            sleep_duration = max(target_frame_duration - frame_duration, 0)  # Calculate the remaining time for the sleep duration
            time.sleep(sleep_duration)

    """
    This function stops the recording thread and saves the recorded frames to a video file.
    While doing so it also paints the performed actions from the action history onto the frames.
    """
    def stop_recording(self):
        time.sleep(0.5)
        self.stop_recording_flag = True
        self.recording_thread.join()

        frame = self._get_frame()
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0
        video_writer = cv2.VideoWriter(self.recording_filename, codec, fps, (frame.shape[1], frame.shape[0]))

        self._print("Frames recorded: ", len(self.frames))

        for frame, frame_timestamp in self.frames:
            frame_interactions = [interaction for interaction in self.action_history if interaction.t <= frame_timestamp]
            for interaction in frame_interactions:
                frames_elapsed = int(fps * (frame_timestamp - interaction.t))
                if frames_elapsed <= 30:
                    alpha = 1 - frames_elapsed / 30
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame = interaction.paint_interaction(frame, alpha)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                
            writable_frame = np.ascontiguousarray(frame)
            video_writer.write(writable_frame)

        video_writer.release()
        self._print("Screen recording saved")


    """
    This function is used to test the methods of the class
    """
    def _test(self):
        print(self.get_app_type() == "web")
        print(self.get_window_size())
        self.start_recording()
        self.click(100, 100)
        self.click(200, 200)
        self.click(300, 300)
        time.sleep(3)
        self.slide(100, 100, 200, 200)
        self.slide(100, 100, 100, 400)
        # print(self.get_gui_state())
        self.save_screenshot("test.png")
        print(self.get_action_history())
        self.stop_recording()
        time.sleep(1)