import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy.stats import multivariate_normal

# from matplotlib.tri import Triangulation

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})

# Create a list of images, mus and sigmas
images = ['screen.png', 'screen.png', 'screen.png']  # Replace with your actual images
mus = [[0.3, 0.3], [0.3, 0.3], [0.3, 0.3]]
sigmas = [50000, 10, 1]

fig = plt.figure(figsize=(6 * len(images), 6))  # Change figure size as needed

for i in range(len(images)):
    MU = mus[i]
    SIGMA = sigmas[i]

    # Read the image with OpenCV
    img = cv2.imread(images[i])
    # Change the color from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image to 256x256
    img = cv2.resize(img, (500, 500))

    # Orgird to store data
    x, y = np.ogrid[0:img.shape[0], 0:img.shape[1]]

    # In Python3 matplotlib assumes rgbdata in range 0.0 to 1.0
    img = img.astype('float32')/255

    # Apply gamma correction to increase brightness
    gamma = 1.5  # change this to adjust the level of brightness
    img = np.power(img, 1/gamma)

    ax = fig.add_subplot(1, len(images), i+1, projection='3d')

    # Plot data
    ax.plot_surface(x, y, np.atleast_2d(0), rstride=1, cstride=1, facecolors=img)

    # Define the mean and covariance matrix for the Gaussian
    mu = np.array([int(MU[0]*img.shape[0]), int(MU[1]*img.shape[1])])
    Sigma = np.array([[int(SIGMA*img.shape[0]), 0], [0, int(SIGMA*img.shape[1])]])

    # Define the grid over which we'll calculate the Gaussian
    X, Y = np.meshgrid(np.linspace(0, img.shape[0], img.shape[0]), np.linspace(0, img.shape[1], img.shape[1]))

    # Calculate the Gaussian over the grid
    Z = multivariate_normal.pdf(np.dstack([X, Y]), mean=mu, cov=Sigma)

    # Display the surface plot
    ax.plot_surface(X, Y, Z, facecolors=None, edgecolors='k', linewidths=0.2, alpha=0.0)

    # Remove axis numbers and gridlines
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)

    ax.xaxis.pane.fill = False  # remove the background for the X axis pane
    ax.yaxis.pane.fill = False  # remove the background for the Y axis pane
    ax.zaxis.pane.fill = False  # remove the background for the Z axis pane

    ax.xaxis.pane.set_edgecolor('white')  # set the x axis grid to white
    ax.yaxis.pane.set_edgecolor('white')  # set the y axis grid to white
    ax.zaxis.pane.set_edgecolor('white')  # set the z axis grid to white

    # Set axis labels with large font size
    ax.set_xlabel('Y', labelpad=-10, fontsize=20)
    ax.set_ylabel('X', labelpad=-10, fontsize=20)
    ax.set_zlabel('Probability', labelpad=-10, fontsize=20)

# Save plot as png 1000x1000 remove white space, custom tight padding
plt.savefig('plot.png', dpi=500, bbox_inches='tight', pad_inches=0.7)
