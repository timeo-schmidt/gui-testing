"""
Preambles allow the manual setting of cookies and actions to be performed when a website is loaded.
e.G. Logging in to a website, accepting cookies etc.
"""

"""
This is the preamble for youtube.com
"""

import time
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

class YouTubePreamble:
    def __init__(self, web_app_interface):
        assert(web_app_interface.get_app_type() == "web")
        self.web_app_interface = web_app_interface

    def _inject_cookies(self):
        self.cookies = [
            # {'domain': '.youtube.com', 'expiry': 1716038156, 'httpOnly': False, 'name': 'PREF', 'path': '/', 'sameSite': 'Lax', 'secure': True, 'value': 'f4=4000000&f6=40000000&tz=Europe.Berlin'},
            # {'domain': '.youtube.com', 'expiry': 1716044602, 'httpOnly': False, 'name': 'CONSENT', 'path': '/', 'sameSite': 'Lax', 'secure': True, 'value': 'YES+cb.20211005-08-p0.de+FX+105'},
            # {'domain': '.youtube.com', 'expiry': 1715606149, 'httpOnly': True, 'name': '__Secure-YEC', 'path': '/', 'sameSite': 'Lax', 'secure': True, 'value': 'CgtCUmktMVNrUmQ2dyiHpOWhBg%3D%3D'},
            # {'domain': '.youtube.com', 'httpOnly': True, 'name': 'YSC', 'path': '/', 'sameSite': 'None', 'secure': True, 'value': 'OOV6w5ReVH4'}
        ]
        for cookie in self.cookies:
            self.web_app_interface.browser.add_cookie(cookie)
    
    def _inject_actions(self):
        time.sleep(3)
        self.web_app_interface.browser.find_element(By.XPATH, "//button[@aria-label='Accept the use of cookies and other data for the purposes described']").click()

        time.sleep(2)
        # Get the anchor tag with title="Trending" and click it
        self.web_app_interface.browser.find_element(By.XPATH, "//a[@title='Trending']").click()
        time.sleep(1)
    
    def inject(self):
        self._inject_cookies()
        if self.web_app_interface.verbose:
            print("Injected cookies for youtube.com")
        self._inject_actions()
        if self.web_app_interface.verbose:
            print("Injected actions for youtube.com")

class RWAPreamble:
    def __init__(self, web_app_interface):
        assert(web_app_interface.get_app_type() == "web")
        self.web_app_interface = web_app_interface

    def _inject_cookies(self):
        self.cookies = []
        for cookie in self.cookies:
            self.web_app_interface.browser.add_cookie(cookie)
    
    def _inject_actions(self):
        # Navigate to the /signup page
        self.web_app_interface.browser.get("http://localhost:3000/signup")
        time.sleep(0.5)
        # Into the input field with #firstName enter "Bob"
        self.web_app_interface.browser.find_element(By.ID, "firstName").send_keys("Bob")
        # Into the input field with #lastName enter "Smith"
        self.web_app_interface.browser.find_element(By.ID, "lastName").send_keys("Smith")
        # Into the input field with #username enter "bob.smith"
        self.web_app_interface.browser.find_element(By.ID, "username").send_keys("bob.smith")
        # Into the input field with #password enter "password"
        self.web_app_interface.browser.find_element(By.ID, "password").send_keys("password")
        # Into the input field with #confirmPassword enter "password" and press enter
        self.web_app_interface.browser.find_element(By.ID, "confirmPassword").send_keys("password")
        # Click on the button with type="submit"
        self.web_app_interface.browser.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
        
        time.sleep(0.5)
        # Now log in
        
        # Into the input field with #username enter "bob.smith"
        self.web_app_interface.browser.find_element(By.ID, "username").send_keys("bob.smith")
        # Into the input field with #password enter "password" and press enter
        self.web_app_interface.browser.find_element(By.ID, "password").send_keys("password")
        self.web_app_interface.browser.find_element(By.ID, "password").send_keys(Keys.ENTER)
        time.sleep(0.5)
    
    def inject(self):
        self._inject_cookies()
        if self.web_app_interface.verbose:
            print("Injected cookies for ", self.web_app_interface.starting_url)
        self._inject_actions()
        if self.web_app_interface.verbose:
            print("Injected actions for ", self.web_app_interface.starting_url)


"""
This function returns the preamble for a given website
"""
def inject_preamble(web_app_interface):
    preamble_map = {
        "https://www.youtube.com/": YouTubePreamble,
        "http://localhost:3000/": RWAPreamble
    }

    if web_app_interface.starting_url in preamble_map:
        preamble_map[web_app_interface.starting_url](web_app_interface).inject()