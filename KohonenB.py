import numpy as np
import cv2
import matplotlib.pyplot as plt
from sompy import SOMFactory

def getDataFromImage(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    points = []
    for i in range(5000):
        x = int(np.random.uniform(0, 100))
        y = int(np.random.uniform(0, 100))
        if img[x][y] < img.max():
                points.append([y, x])
    return points

# scatter data
data = getDataFromImage('hand_wo_finger.png')
data = np.array(data)
data = data / 100
data[:, 1] = 1 - data[:, 1]
plt.scatter(data[:, 0], data[:, 1])
plt.show()

map_size = [20, 20]  # Size of the SOM grid
som = SOMFactory.build(data, map_size, mask=None, mapshape='planar', lattice='rect')

# Initialize a figure for visualization
fig = plt.figure()

# Train the SOM
for i in range(1000):  # Perform 1000 iterations
    som.train(1)  # Train for one iteration

    # Get the trained SOM grid
    som_grid = som._normalizer.denormalize_by(som.data_raw, som.codebook.matrix)

    # Plot the SOM grid
    plt.clf()
    plt.imshow(som_grid, origin='lower')
    plt.title(f'Iteration {i+1}')
    plt.colorbar()
    plt.pause(0.01)  # Pause for a short time to display the plot

plt.show()  # Show the final plot

