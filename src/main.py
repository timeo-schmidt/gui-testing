from app_interface.web_app_interface import WebAppInterface
from exploration_algorithms.visual_rl import VisualRL

if __name__ == "__main__":
    web = WebAppInterface(detached=False, screen_size=(500,500))
    web._test()