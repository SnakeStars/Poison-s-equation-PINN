# Import Library


import matplotlib.pyplot as plt
import numpy as np

# Define Data

x = np.linspace(0, 10, 100)
y = np.sin(x)

# Set color map

plt.scatter(x, y, c=y, cmap='Set2')

# Display

plt.show()