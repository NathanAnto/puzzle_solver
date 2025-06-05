import json
import os
from PIL import Image
import matplotlib.pyplot as plt

# Load JSON from file
with open('puzzle_solution_dfs.json', 'r') as f:
    data = json.load(f)

grid_rows, grid_cols = data['grid_size']
pieces = data['pieces']

# Create a figure with subplots arranged as grid_size
fig, axes = plt.subplots(grid_rows, grid_cols)

# If only one row/col, axes may not be 2D array; force it to be
if grid_rows == 1 or grid_cols == 1:
    axes = axes.reshape(grid_rows, grid_cols)

# Plot each piece
for piece in pieces:
    row = piece['position']['row']
    col = piece['position']['col']
    filename = os.path.join('piece', piece['filename'])
    rotation = piece['rotation']

    # Load and rotate the image
    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found.")
        continue

    img = Image.open(filename).rotate(-rotation, expand=True)  # PIL rotates counter-clockwise

    # Plot on the correct subplot
    ax = axes[row][col]
    ax.imshow(img)
    ax.axis('off')  # Turn off axes for a cleaner display

plt.tight_layout()
plt.show()