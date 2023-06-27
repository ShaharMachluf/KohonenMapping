import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom

# Generate sample data
data = np.random.rand(100, 10)  # 100 samples with 10 features

# Define SOM parameters
map_size = (20, 20)  # Size of the SOM grid
input_len = data.shape[1]  # Number of input features

# Initialize the SOM
som = MiniSom(map_size[0], map_size[1], input_len, sigma=1.0, learning_rate=0.5)

# Initialize the weights with random values
som.random_weights_init(data)

# Initialize a figure for visualization
fig = plt.figure()

# Train the SOM
for i in range(1000):  # Perform 1000 iterations
    som.update(data[i % len(data)], som.winner(data[i % len(data)]), i, 1000)  # Update the SOM with a data sample

    # Plot the SOM grid
    plt.clf()
    plt.imshow(som.get_weights().T, origin='lower')
    plt.title(f'Iteration {i+1}')
    plt.colorbar()
    plt.pause(0.01)  # Pause for a short time to display the plot

plt.show()  # Show the final plot
