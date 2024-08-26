import matplotlib.pyplot as plt
import numpy as np

from imports import *

# Define the indices
x = list(range(12))

# Define the first variable with 12 values
y1 = [3, 5, 2, 6, 7, 8, 5, 4, 7, 8, 5, 6]

# Define the second variable with values at indices 2, 6, and 10
# Use np.nan for the indices with no values
y2 = [
    np.nan,
    np.nan,
    10,
    np.nan,
    np.nan,
    np.nan,
    15,
    np.nan,
    np.nan,
    np.nan,
    12,
    np.nan,
]

# Plot the first variable
plt.plot(x, y1, label="y1 (12 values)", marker="o")

# Plot the second variable
plt.plot(x, y2, label="y2 (3 values)", marker="o")

# Add labels and legend
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Line Graph with Different Data Points")
plt.legend()

# Show the plot
plt.show()
