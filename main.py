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


dataset = np.random.rand(1000, 2)

# dataset[:, 1] = dataset[:, 0] + 0.1 * np.random.randn(1000) - 0.05  #  y = x +/- noise

# y is normal distribution
# iter = 5
# for i in range(iter):
#     random_numbers = np.random.randn(1000)
#     normalized_numbers = (random_numbers - random_numbers.min()) / (random_numbers.max() - random_numbers.min())
#
#     dataset[:, 1] += normalized_numbers
# dataset[:, 1] = dataset[:, 1] / iter

grid_size = 20
learning_rate = 0.1
neighborhood_radius = 5
num_iterations = 100

weights = kohonen_algorithm(dataset, grid_size, learning_rate, neighborhood_radius, num_iterations)

sorted_indices = np.argsort(weights[:, 0])
sorted_weights = weights[sorted_indices]

print("Sorted Weight Vectors:")
print(sorted_weights)

dataset_x = dataset[:, 0]
dataset_y = dataset[:, 1]
weights_x = sorted_weights[:, 0]
weights_y = sorted_weights[:, 1]

plt.scatter(dataset_x, dataset_y, color='blue', label='Dataset')
plt.scatter(weights_x, weights_y, color='red', label='Weights')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Dataset and Weights')
plt.legend()
plt.show()
