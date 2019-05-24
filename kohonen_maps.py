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
  #plt.scatter(*zip(*map), c="red")
  #plt.axis([0, 1, 0, 1])
  #plt.show()
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
def kohonen_maps(input_data, nb_representants, plot = False):
    # Définir les coordonnées des Wi sur 0 - 1 pour les x et y
    map = build_map(nb_representants)

    # Prendre au hasard les features dans les données
    w_i = put_features_with_data(input_data, map)

    # Variables variables
    batch_size = 1
    alpha = 0.5
    gamma = 0.5
    iterations_max = 1

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
        if plot:
            plt.scatter(*zip(*input_data), c="blue")
            plt.scatter(*zip(*w_i), c="red")
            plt.show()
        print("===Iteration===")
        print(iteration)
        iteration = 1+iteration
    return w_i

#x_tt = [[25, 79], [34, 51], [22, 53], [27, 78], [33, 59], [33, 74], [31, 73], [22, 57], [35, 69], [34, 75], [67, 51], [54, 32], [57, 40], [43, 47], [50, 53], [57, 36], [59, 35], [52, 58], [65, 59], [47, 50], [49, 25], [48, 20], [35, 14], [33, 12], [44, 20], [45, 5], [38, 29], [43, 27], [51, 8], [46, 7]]
#rep = kohonen_maps(x_tt, 3)
#plt.scatter(*zip(*x_tt), c="blue")
#plt.scatter(*zip(*rep), c="red")
#plt.axis([0, 1, 0, 1])
#plt.show()

#print(np.array(x_tt))

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


#Essai sur MNIST
#representatives = kohonen_maps(x_train[:1000], 15)
#plt.figure(figsize=(40, 40))
#for i in range(len(representatives)):
      #display encoded images
#      ax = plt.subplot(21, 20, i +1 )
#     plt.imshow(representatives[i].reshape(28, 28))
#      plt.gray()
#      ax.get_xaxis().set_visible(False)
#      ax.get_yaxis().set_visible(False)
#plt.show()


#Representation 2D
#rep = kohonen_maps(x_train[:1000], 2, False)
#coordinates = []
#for i in x_train[:1000]:
#    coordinates.append([euclidian_dist(i, rep[0]), euclidian_dist(i, rep[1])])
#plt.scatter(*zip(*coordinates), c=y_train[:1000])
#plt.show()

def random_sample(data, nb_data):
    rds = []
    sample = []
    for i in range(nb_data):
        rd = random.randint(0, len(data) - 1)
        check_if_exists = True
        while check_if_exists:
            if rd not in rds:
                check_if_exists = False
                sample.append(data[rd])
                rds.append(rd)
            rd = random.randint(0, len(data) - 1)
    rds.clear()
    return sample


#Essai Dataset choisi : fruits
x_fruits = np.load("../fruits/fruits-360/test_images28.npy")
x_fruits = x_fruits.astype('float32') / 255.
x_fruits = x_fruits.reshape((len(x_fruits), np.prod(x_fruits.shape[1:])))
print("Normalize and flatten data...")
print("x_fruits shape : " + str(x_fruits.shape))
_fruits = random_sample(x_fruits, 1000)

nb_rep = 15
representatives = kohonen_maps(x_fruits, nb_rep)
map = build_map(nb_rep)
#plt.scatter(*zip(*map), c="red")
#plt.axis([0, 1, 0, 1])
#plt.show()


plt.figure(figsize=(40, 40))
for i in range(len(representatives)):
      #display encoded images
      ax = plt.subplot(21, 20, i +1 )
      plt.imshow(representatives[i].reshape(28, 28), extent=[map[i][0],map[i][1], map[i][0],map[i][1]])
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
plt.show()

