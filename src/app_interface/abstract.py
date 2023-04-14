import time
from abc import ABC, abstractmethod


"""
This is the abstract class that all app interfaces will inherit from.
"""
class AbstractAppInterface(ABC):

    # Setup the App

    def __init__(self, screen_size=(1920, 1080), screenshot_dir="./screenshots"):
        super().__init__()

    # Get App Properties

    """
    Returns the type of app that this interface is for as a string
    """
    @abstractmethod
    def get_app_type(self):
        pass

    """
    Returns the pixel size of the app window as a tuple (x,y)
    """
    @abstractmethod
    def get_window_size(self):
        pass


    # Perform App Actions

    """
    Moves the mouse to the specified pixel location
    """
    @abstractmethod
    def move_mouse(self, x, y):
        pass
    
    """
    Clicks at the specified pixel location
    """
    @abstractmethod
    def click(self, x, y):
        pass

    """
    Scrolls/Slides the app window by the specified pixel amount with a start and end location
    """
    @abstractmethod
    def slide(self, x, y, x2, y2):
        pass


    # Get App State and interaction history

    """
    Returns information about the current state of the app as a dictionary
    """
    @abstractmethod
    def get_gui_state(self):
        pass

    """
    Tries to fix deadends in the app by injecting actions manually
    """
    @abstractmethod
    def fix_deadends(self):
        pass

    """
    Returns a screenshot of the app window as a PIL image
    """
    @abstractmethod
    def get_screenshot(self):
        pass

    """
    Saves a screenshot of the app window as a PNG file on the computer
    """
    @abstractmethod
    def save_screenshot(self, filename=time.time()):
        pass

    """
    Gets the interaction history of the app as a list of dictionaries
    """
    @abstractmethod
    def get_action_history(self):
        pass

    # Debugging and Utility Tools

    """
    Start screen recording
    """
    @abstractmethod
    def start_recording(self):
        pass

    """
    Stop screen recording
    """
    @abstractmethod
    def stop_recording(self):
        pass

