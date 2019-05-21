import random
import sys

import numpy as np
import time
import matplotlib.pyplot as plt
from keras.datasets import mnist

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

x_tt = [(1,1), (0, 2), (3, 2), (5,1), (4, 2), (6, 2), (1, -1), (0, -2), (3, -2)]

plt.scatter(*zip(*x_tt))
plt.show()

def euclidian_dist(input, centre):
    sum = 0
    for i in range(len(input)):
        sum += pow(input[i] + centre[i], 2)
    return np.sqrt(sum)

def lloyd_algorithm(input_data, nb_Representatives):
    iterations = 0
    representatives = {}
    iterations_max = 100

    # Initialisation des centres
    for i in range(nb_Representatives):
        check_if_already_present = True
        rd = random.randint(0, len(input_data)-1)
        rds = {}
        while(check_if_already_present):
            if(rd not in rds):
                rds[i] = rd
                representatives[i] = input_data[rd]
                check_if_already_present = False
            rd = random.randint(0, len(input_data)-1)

    print("FIRST REP")
    print(representatives)
    # Répétition de l'algorithme
    old_clusters = {}
    while iterations < iterations_max:
        clusters = {}
        dict_dist = {}
        # Allocation au centre le plus proche
        for i in range(len(input_data)):
            for j in range(len(representatives)):
                dict_dist[euclidian_dist(input_data[i],representatives[j])] =  j
            clusters[i] = dict_dist[min(dict_dist.keys())]

        #Recalcul des centres
        sum_clusters = {}
        for i in range(len(clusters)):
            sum_clusters[i] = {}
            for j in range(len(input_data[i])):
                sum_clusters[i][j] = 0
        for i in range(len(clusters)):
            for j in range(len(input_data[i])):
                sum_clusters[clusters[i]][j] += input_data[i][j]

        old_representatives = representatives
        for i in range(len(sum_clusters)):
            for j in range(len(sum_clusters[i])):
                representatives[i][j] = sum(sum_clusters[i].values()) / len(sum_clusters[i])
        print("NEW REP")
        print(representatives)
        # Comparaison ancien resultat / nouveau
        if old_clusters == clusters or old_representatives == representatives:
            break
        old_clusters = clusters

lloyd_algorithm(x_tt, 3)





