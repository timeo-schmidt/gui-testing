"""
This script allows the recording of human interactions with the environment to collect data for a baseline.
"""
import matplotlib.pyplot as plt
from PIL import Image
import time
import numpy as np

def get_action(obs, env):
    # Allow the page to load
    time.sleep(0.2)

    # Get the screenshot
    img = env.web_app_interface.get_screenshot()

    # Show the grayscale image
    plt.figure(figsize = (18,7))
    plt.imshow(img, cmap='gray')
    
    clicked_coordinates = []
    
    # Function to handle click events
    def onclick(event):
        # Scale the coordinates to be between -1 and 1
        print(event.xdata, event.ydata)
        ix, iy = event.xdata / img.width * 2 - 1,  event.ydata / img.height * 2 - 1
        print(ix, iy)
        print('You clicked on coordinates:', ix, iy)
        clicked_coordinates.append((ix, iy))
        plt.close()  # Close the image after the click
    
    # Connect the function with the click event
    cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)

    # We use a blocking call to plt.show() here to "wait" for user input.
    plt.show(block=True)

    x = clicked_coordinates[-1][0]
    y = clicked_coordinates[-1][1]
    
    # return the last clicked location (or None if no clicks were made)
    return x,y