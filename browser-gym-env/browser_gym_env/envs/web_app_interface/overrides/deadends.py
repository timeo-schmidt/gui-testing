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
    def __init__(self, web_app_interface, starting_url):
        assert(web_app_interface.get_app_type() == "web")
        self.web_app_interface = web_app_interface
        self.starting_url = starting_url

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
            
            return True
        # Check if there are any additional tabs that need to be closed
        elif len(self.web_app_interface.browser.window_handles) > 1:
            # Close all windows except the original
            for handle in self.web_app_interface.browser.window_handles:
                if handle != self.web_app_interface.original_tab_handle:
                    self.web_app_interface.browser.switch_to.window(handle)
                    self.web_app_interface.browser.close()
            self.web_app_interface.browser.switch_to.window(self.web_app_interface.original_tab_handle)
            # Navigate back to the home URL
            self.web_app_interface.browser.get(self.web_app_interface.starting_url)
            if self.web_app_interface.verbose:
                print("Detected Extra Tabs. Closing Extra Tabs and Resetting.")
            time.sleep(1)
            return True
        else:
            return False
            


"""
This function returns the preamble for a given website
"""
def recover_deadend(web_app_interface):
    preamble_map = {
        "http://localhost:3000/": RWADeadend
    }

    if web_app_interface.starting_url in preamble_map:
        return preamble_map[web_app_interface.starting_url](web_app_interface, web_app_interface.starting_url).recover_deadend()