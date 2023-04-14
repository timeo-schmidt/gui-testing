from exploration_algorithms.visual_rl import VisualRL
from app_interface.web_app_interface import WebAppInterface

if __name__ == "__main__":
    app_interface = WebAppInterface(detached=False, starting_url="http://localhost:3000/")
    app_interface.start_recording()
    visual_rl = VisualRL(app_interface)
    visual_rl.explore(200)
    app_interface.stop_recording()