import numpy as np
import matplotlib.pyplot as plt
from sompy import SOMFactory

# Generate sample data
data = np.random.rand(100, 10)  # 100 samples with 10 features

# Define SOM parameters
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
