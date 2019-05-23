import math
import random
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from keras.datasets import mnist


def build_map(nb_representants):
  map = []
  step = 1 / (nb_representants + 1)
  for i in np.arange(step, 1 - step/2, step):
    for j in np.arange(step, 1 - step/2, step):
      map.append([i, j])
  plt.scatter(*zip(*map), c="red")
  plt.axis([0, 1, 0, 1])
  plt.show()
  return map

def put_features_with_data(input_data, map):
  w_i = []
  for i in range(len(map)):
    rd = random.randint(0, len(input_data) - 1)
    w_i.append(input_data[rd])
  return w_i


def choose_sample_in_data(input_data):
  rd = random.randint(0, len(input_data) -1)
  return input_data[rd]

def euclidian_dist(input, centre):
    sum = 0
    for i in range(len(input)):
        sum += pow(input[i] - centre[i], 2)
    return np.sqrt(sum)

def find_closest_neuron(input, w_i):
  distances = {}
  j = 0
  for i in w_i:
    distances[euclidian_dist(input, i)] = j
    j = j + 1
  return w_i[distances[min(distances.keys())]]

def modify_feature_vector(coordinates_w, coordinates_neighbour, feature, sample, alpha, gamma):
  new_feature = feature + alpha * math.exp(-(np.linalg.norm(euclidian_dist(coordinates_w, coordinates_neighbour)))/ 2 * gamma) * np.subtract(np.array(sample), np.array(feature))
  return new_feature

def find_neighbours(w_i, representative):
  step = 1 / len(w_i)
  neighbours = []
  for i in w_i:
    for j in representative:
      if j + step in i or j - step in i:
        neighbours.append(i)
  return neighbours

def find_representative_index_by_value(w_i, best_representative):
    for i in range(len(w_i)):
        if np.array_equal(best_representative, w_i[i]):
            return i

# NB représentants = nb representants sur x et sur y
def kohonen_maps(input_data, nb_representants):
    # Définir les coordonnées des Wi sur 0 - 1 pour les x et y
    map = build_map(nb_representants)

    # Prendre au hasard les features dans les données
    w_i = put_features_with_data(input_data, map)

    # Variables variables
    batch_size = 1
    alpha = 0.5
    gamma = 0.5
    iterations_max = 100

    # Repeat jusqu'a iteration max
    iteration = 0
    while iteration < iterations_max:
        # Prendre un input data au hasard
        sample = choose_sample_in_data(input_data)

        # Trouver son meilleur représentant
        best_representative = find_closest_neuron(sample, w_i)

        # Trouver index best_representative
        index = find_representative_index_by_value(w_i, best_representative)

        # Modifier le feature vector choisi avec formule
        w_i[index] = modify_feature_vector(map[index], map[index], w_i[index], sample, alpha, gamma)

        # Modifier les features des voisins avec formule
        neighbours = find_neighbours(w_i, best_representative)
        for i in neighbours:
            w_i[i] = modify_feature_vector(map[find_representative_index_by_value(w_i, i)], map[index], sample, alpha, gamma)
        plt.scatter(*zip(*input_data), c="blue")
        plt.scatter(*zip(*w_i), c="red")
        plt.show()
        iteration = ++iteration
    return w_i

x_tt = [[25, 79], [34, 51], [22, 53], [27, 78], [33, 59], [33, 74], [31, 73], [22, 57], [35, 69], [34, 75], [67, 51], [54, 32], [57, 40], [43, 47], [50, 53], [57, 36], [59, 35], [52, 58], [65, 59], [47, 50], [49, 25], [48, 20], [35, 14], [33, 12], [44, 20], [45, 5], [38, 29], [43, 27], [51, 8], [46, 7]]
rep = kohonen_maps(x_tt, 3)
plt.scatter(*zip(*x_tt), c="blue")
plt.scatter(*zip(*rep), c="red")
plt.axis([0, 1, 0, 1])
plt.show()
#print(np.array(x_tt))