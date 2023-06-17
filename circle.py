import numpy as np
import matplotlib.pyplot as plt

def initialize_weights(grid_size, input_dim):
    weights = np.random.rand(grid_size, input_dim)
    return weights

def find_bmu(input_vector, weights):
    distances = np.linalg.norm(weights - input_vector, axis=1)
    bmu_index = np.argmin(distances)
    return bmu_index

def update_weights(bmu_index, input_vector, weights, learning_rate, neighborhood_radius):
    influence = np.exp(-np.arange(weights.shape[0]) ** 2 / (2 * neighborhood_radius ** 2))

    for i in range(weights.shape[0]):
        distance = np.abs(i - bmu_index)
        if distance <= neighborhood_radius:
            delta = learning_rate * influence[distance] * (input_vector - weights[i])
            weights[i] += delta

def kohonen_algorithm(dataset, grid_size, learning_rate, neighborhood_radius, num_iterations):
    input_dim = dataset.shape[1]
    weights = initialize_weights(grid_size, input_dim)
    weights = weights - 0.5
    weights = weights * 2

    for iteration in range(num_iterations):
        np.random.shuffle(dataset)
        # if iteration % 10 == 0:
        dataset_x = dataset[:, 0]
        dataset_y = dataset[:, 1]
        weights_x = weights[:, 0]
        weights_y = weights[:, 1]

        plt.scatter(dataset_x, dataset_y, color='blue', label='Dataset')
        plt.scatter(weights_x, weights_y, color='red', label='Weights')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Scatter Plot of Dataset and Weights' + ' (Iteration: ' + str(iteration) + ')')
        plt.legend()
        plt.show()
        for input_vector in dataset:
            bmu_index = find_bmu(input_vector, weights)

            update_weights(bmu_index, input_vector, weights, learning_rate, neighborhood_radius)


    return weights

points = []
while len(points) < 1000:
    x = np.random.uniform(-4, 4)
    y = np.random.uniform(-4, 4)
    distance_squared = x**2 + y**2
    if 4 <= distance_squared <= 16:
        points.append((x, y))

dataset = np.array(points)

grid_size = 300
learning_rate = 0.1
neighborhood_radius = 10
num_iterations = 100
weights = kohonen_algorithm(dataset, grid_size, learning_rate, neighborhood_radius, num_iterations)