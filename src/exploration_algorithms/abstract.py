from abc import ABC, abstractmethod

"""
This is the abstract class that all exploration algorithms will inherit from.
"""
class AbstractExplorationAlgorithm(ABC):

    # Setup the Exploration Algorithm

    def __init__(self, app_interface):
        super().__init__()

    # Explore the App

    @abstractmethod
    def explore(self, app_interface):
        pass