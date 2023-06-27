import numpy as np
import cv2
import matplotlib.pyplot as plt
from minisom import MiniSom

def initialize_weights(grid_size, input_dim):
    grid_size = 20  # Number of points along each dimension

    # Generate the grid of points
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)

    # Reshape the grid points into a 2D array
    weights = np.column_stack((xx.ravel(), yy.ravel()))
    return weights

def find_bmu(input_vector, weights):
    distances = np.linalg.norm(weights - input_vector, axis=1)
    bmu_index = np.argmin(distances)
    return bmu_index

def update_weights(bmu_index, input_vector, weights, learning_rate, neighborhood_radius):
    # influence = np.exp(-np.arange(weights.shape[0]) ** 2 / (2 * neighborhood_radius ** 2))

    for i in range(weights.shape[0]):
        distance = np.linalg.norm(weights[bmu_index]-weights[i])
        influence = np.exp((-distance**2)/2*neighborhood_radius**2) / neighborhood_radius**2
        if distance <= neighborhood_radius:
            delta = learning_rate * influence * (input_vector - weights[i])
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
        # neighborhood_radius = neighborhood_radius * np.exp(-iteration/lamda)


    return weights

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

# dataset[:, 1] = dataset[:, 0] + 0.1 * np.random.randn(1000) - 0.05  #  y = x +/- noise

# y is normal distribution
# iter = 5
# for i in range(iter):
#     random_numbers = np.random.randn(1000)
#     normalized_numbers = (random_numbers - random_numbers.min()) / (random_numbers.max() - random_numbers.min())
#
#     dataset[:, 1] += normalized_numbers
# dataset[:, 1] = dataset[:, 1] / iter

grid_size = 400
learning_rate = 0.01
neighborhood_radius = 7
num_iterations = 100
lamda = 3
weights = kohonen_algorithm(data, grid_size, learning_rate, neighborhood_radius, num_iterations)





# # Define SOM parameters
# map_size = (20, 20)  # Size of the SOM grid
# input_len = data.shape[1]  # Number of input features
#
# # Initialize the SOM
# som = MiniSom(map_size[0], map_size[1], input_len, sigma=1.0, learning_rate=0.5)
#
# # Initialize the weights with random values
# som.random_weights_init(data)
#
# # Initialize a figure for visualization
# fig, axs = plt.subplots(map_size[0], map_size[1])
#
# # Train the SOM
# for i in range(1000):  # Perform 1000 iterations
#     som.update(data[i % len(data)], som.winner(data[i % len(data)]), i, 1000)  # Update the SOM with a data sample
#
#     # Plot the SOM grid
#     for x in range(map_size[0]):
#         for y in range(map_size[1]):
#             axs[x, y].imshow(som.get_weights()[x, y].reshape(-1, 1), cmap='gray')
#             axs[x, y].axis('off')
#     plt.title(f'Iteration {i+1}')
#     plt.colorbar()
#     plt.pause(0.01)  # Pause for a short time to display the plot
#
# plt.show()  # Show the final plot
#
