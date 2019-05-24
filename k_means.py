import random
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from keras.datasets import mnist

def euclidian_dist(input, centre):
    sum = 0
    for i in range(len(input)):
        sum += pow(input[i] - centre[i], 2)
    return np.sqrt(sum)

def lloyd_algorithm(input_data, nb_Representatives, plot):
    iterations = 0
    representatives = np.zeros([nb_Representatives, len(input_data[0])])
    iterations_max = 100

    # Initialisation des centres
    rds = []
    for i in range(nb_Representatives):
        rd = random.randint(0, len(input_data) - 1)
        check_if_exists = True
        while check_if_exists:
            if rd not in rds:
                check_if_exists = False
                representatives[i] = input_data[rd]
                rds.append(rd)
            rd = random.randint(0, len(input_data) -1)
    rds.clear()
    if plot:
        plt.scatter(*zip(*input_data))
        plt.scatter(*zip(*representatives), c="red")
        plt.show()

    # Répétition de l'algorithme
    while iterations < iterations_max:

        # Allocation au centre le plus proche
        clusters = {}
        distances = {}
        for i in range(len(input_data)):
            for j in range(len(representatives)):
                distances[euclidian_dist(input_data[i], representatives[j])] = j
            clusters[i] = distances[min(distances.keys())]
            distances.clear()

        #Recalcul des centres
        sum_clusters = np.zeros([nb_Representatives, len(representatives[0]) + 1])
        for i in range(len(input_data)):
            for j in range(len(input_data[0])):
                sum_clusters[clusters[i]][j] += input_data[i][j]
            sum_clusters[clusters[i]][len(representatives[0])] += 1

        old_representative = representatives
        representatives = np.zeros([nb_Representatives, len(input_data[0])])
        for i in range(len(sum_clusters)):
            for j in range(len(sum_clusters[0])-1):
                if sum_clusters[i][j] != 0:
                    if sum_clusters[i][len(sum_clusters[i]) - 1] != 0:
                        representatives[i][j] = sum_clusters[i][j] / sum_clusters[i][len(sum_clusters[i]) - 1]

        if plot:
            plt.scatter(*zip(*input_data))
            plt.scatter(*zip(*representatives), c="red")
            plt.show()

        # Comparaison ancien resultat / nouveau
        if (representatives == old_representative).all():
            print("BREAK !!!")
            return representatives

        print("ITERATION ====== ")
        print(iterations)
        iterations += 1
    return  representatives

# Loading data from Mnist
print("Loading data from Mnist...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalise data
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Flatten the data to have 784 inputs instead of 28x28
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print("Normalize and flatten data...")
print("x_train shape : " + str(x_train.shape) + " | y_train shape : " + str(y_train.shape))
print("x_test shape : " + str(x_test.shape) + " | y_test shape : " + str(y_test.shape))

x_tt = [[25, 79], [34, 51], [22, 53], [27, 78], [33, 59], [33, 74], [31, 73], [22, 57], [35, 69], [34, 75], [67, 51], [54, 32], [57, 40], [43, 47], [50, 53], [57, 36], [59, 35], [52, 58], [65, 59], [47, 50], [49, 25], [48, 20], [35, 14], [33, 12], [44, 20], [45, 5], [38, 29], [43, 27], [51, 8], [46, 7]]

x_fruits = np.load("test_images28.npy")
x_fruits = x_fruits.astype('float32') / 255.
x_fruits = x_fruits.reshape((len(x_fruits), np.prod(x_fruits.shape[1:])))
print("Normalize and flatten data...")
print("x_fruits shape : " + str(x_fruits.shape))

lloyd_algorithm(x_fruits,103, True)

# representatives = lloyd_algorithm(x_train[:1000], 12, False)
# plt.figure(figsize=(40, 4))
# for i in range(12):
#     # display encoded images
#     ax = plt.subplot(3, 20, i + 1 + 20)
#     plt.imshow(representatives[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()



