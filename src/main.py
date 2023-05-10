from app_interface.web_app_interface import WebAppInterface
from exploration_algorithms.visual_rl import VisualRL

if __name__ == "__main__":
    web = WebAppInterface(detached=False, starting_url="http://localhost:3000/")
    web._test()
    # print(web.browser.get_cookies())