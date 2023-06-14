import numpy as np


# Function to initialize the weight vectors
def initialize_weights(grid_size, input_dim):
    weights = np.random.rand(grid_size, input_dim)
    return weights


# Function to find the Best Matching Unit (BMU)
def find_bmu(input_vector, weights):
    distances = np.linalg.norm(weights - input_vector, axis=1)
    bmu_index = np.argmin(distances)
    return bmu_index


# Function to update the weights of the BMU and its neighbors
def update_weights(bmu_index, input_vector, weights, learning_rate, neighborhood_radius):
    # Compute the influence using a Gaussian neighborhood function
    influence = np.exp(-np.arange(weights.shape[0]) ** 2 / (2 * neighborhood_radius ** 2))

    # Update the BMU and its neighbors
    for i in range(weights.shape[0]):
        distance = np.abs(i - bmu_index)
        if distance <= neighborhood_radius:
            delta = learning_rate * influence[distance] * (input_vector - weights[i])
            weights[i] += delta


# Kohonen algorithm
def kohonen_algorithm(dataset, grid_size, learning_rate, neighborhood_radius, num_iterations):
    input_dim = dataset.shape[1]
    weights = initialize_weights(grid_size, input_dim)

    for iteration in range(num_iterations):
        # Randomly shuffle the dataset
        np.random.shuffle(dataset)

        for input_vector in dataset:
            # Find the BMU
            bmu_index = find_bmu(input_vector, weights)

            # Update the weights of the BMU and its neighbors
            update_weights(bmu_index, input_vector, weights, learning_rate, neighborhood_radius)

    return weights


# Generate a random square dataset
dataset = np.random.rand(1000, 2)

# Set the parameters for the Kohonen algorithm
grid_size = 20
learning_rate = 0.1
neighborhood_radius = 5
num_iterations = 100

# Apply the Kohonen algorithm
weights = kohonen_algorithm(dataset, grid_size, learning_rate, neighborhood_radius, num_iterations)

# Sort the weight vectors based on their position in the line
sorted_indices = np.argsort(weights[:, 0])
sorted_weights = weights[sorted_indices]

# Print the sorted weight vectors
print("Sorted Weight Vectors:")
print(sorted_weights)
