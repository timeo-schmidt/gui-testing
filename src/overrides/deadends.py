"""
Deadends are states of the GUI that are difficult to recover from. (e.g.: Logging out)
"""

"""
This is the preamble for youtube.com
"""

import time
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

class RWADeadend:
    def __init__(self, web_app_interface):
        assert(web_app_interface.get_app_type() == "web")
        self.web_app_interface = web_app_interface

    def recover_deadend(self):
        # Check if the path contains the /signin
        if self.web_app_interface.browser.current_url.endswith("/signin"):
            # If it is, then log in
            # Into the input field with #username enter "bob.smith"
            self.web_app_interface.browser.find_element(By.ID, "username").send_keys("bob.smith")
            # Into the input field with #password enter "password" and press enter
            self.web_app_interface.browser.find_element(By.ID, "password").send_keys("password")
            self.web_app_interface.browser.find_element(By.ID, "password").send_keys(Keys.ENTER)

            if self.web_app_interface.verbose:
                print("Injected actions for ", self.web_app_interface.starting_url)


"""
This function returns the preamble for a given website
"""
def recover_deadend(web_app_interface):
    preamble_map = {
        "http://localhost:3000/": RWADeadend
    }

    if web_app_interface.starting_url in preamble_map:
        preamble_map[web_app_interface.starting_url](web_app_interface).recover_deadend()