import numpy as np
import cv2
import matplotlib.pyplot as plt

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


