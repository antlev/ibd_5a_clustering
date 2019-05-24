import random
import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def covariance_matrix(data):
    centered = (data - np.mean(data.T,axis=1).T)
    return np.matmul(centered.T, centered) / (data.shape[0] - 1)

def euclidian_dist(input, centre):
    sum = 0
    for i in range(len(input)):
        sum += pow(input[i] - centre[i], 2)
    return np.sqrt(sum)

def lloyd_algorithm(input_data,y, nb_Representatives, plot):
    iterations = 0
    representatives = np.zeros([nb_Representatives, len(input_data[0])])
    iterations_max = 10

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
        plt.scatter(*zip(*representatives), c="red", marker='x')
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

        # if plot:
        #     plt.scatter(*zip(*input_data))
        #     plt.scatter(*zip(*representatives), c="red")
        #     plt.show()

        # Comparaison ancien resultat / nouveau
        if (representatives == old_representative).all():
            if plot:
                plt.scatter(*zip(*input_data), c=y)
                plt.scatter(*zip(*representatives), c="red")
                plt.show()
            print("BREAK !!!")
            return representatives

        print("ITERATION ====== ")
        # print(iterations)
        # if plot:
        #     plt.scatter(*zip(*input_data), c=y, marker="x")
        #     plt.scatter(*zip(*representatives), c="red")
        #     plt.show()
        iterations += 1
    if plot:
        plt.scatter(*zip(*input_data), c=y)
        plt.scatter(*zip(*representatives), c="red", marker="x")
        plt.show()
    return  representatives


def my_pca(data, n_components=2):
    if len(data.shape) > 2:
        data = data.reshape((data.shape[0], -1))
    cov = covariance_matrix(data)
    eigen_value, eigen_vectors = eigh(cov)
    # Sorting eigen_vectors using eiugen_values
    order = (-eigen_value).argsort() # Recover the indices if the array were sorted
    eigen_vectors_sorted = np.transpose(eigen_vectors)[order]  # numpy returns the transpose, so we transpose it back
    selected_eigen_vectors_sorted = eigen_vectors_sorted[:n_components]
    selected_eigen_vectors_sorted = np.transpose(selected_eigen_vectors_sorted)
    pca = np.matmul(data, selected_eigen_vectors_sorted)
    return pca, eigen_value[order][:n_components]

def test_pca():
    test = np.array([[random.random(), random.random(), random.random()],[random.random(), random.random(), random.random()],[random.random(), random.random(), random.random()]])
    centered_test = test - (np.mean(test.T, axis=1)).T
    print((np.mean(centered_test.T, axis=1)).T)
    print("Our result : ")
    print(my_pca(centered_test))
    print("SkLearn result : ")
    sklearn_pca = PCA(n_components=2).fit(centered_test)
    print(sklearn_pca.transform(centered_test))
    print(sklearn_pca.singular_values_ )


def test_covariance():
    test = np.array([[random.random(), random.random(), random.random()],[random.random(), random.random(), random.random()],[random.random(), random.random(), random.random()]])
    print("Our result : ")
    print(covariance_matrix(test))
    print("numpy result : ")
    print(np.cov(test))

def all_test():
    test_covariance()
    test_pca()
    test_pca_mnist_2()
    test_pca_mnist()
    test_pca_fruit_dataset

def test_pca_mnist():
    import numpy as np, time, matplotlib.pyplot as plt
    from keras.datasets import mnist
    print("Loading data from Mnist...")
    (X, Y), (_, _) = mnist.load_data()
    # Normalise data
    X = X.astype('float32') / 255.
    # Flatten the data to have 784 inputs instead of 28x28
    # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    print("Normalize and flatten data...")
    print("x_train shape : " + str(X.shape) + " | y_train shape : " + str(Y.shape))
    dataset = np.array(X)
    res, res2 = my_pca(dataset)
    plt.scatter(res[:,0],res[:,1], c=Y)
    plt.show()

def test_generate_pca_mnist():
    import numpy as np, time, matplotlib.pyplot as plt
    from keras.datasets import mnist
    print("Loading data from Mnist...")
    (X, Y), (_, _) = mnist.load_data()
    # Normalise data
    X = X.astype('float32') / 255.
    # Flatten the data to have 784 inputs instead of 28x28
    # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    print("Normalize and flatten data...")
    print("x_train shape : " + str(X.shape) + " | y_train shape : " + str(Y.shape))
    dataset = np.array(X)
    res, res2 = my_pca(dataset)
    plt.scatter(res[:,0],res[:,1], c=Y)
    plt.show()

def test_pca_fruit_dataset():
    x_fruits = np.load("test_images28.npy")
    x_fruits = x_fruits.astype('float32') / 255.
    x_fruits = x_fruits.reshape((len(x_fruits), np.prod(x_fruits.shape[1:])))
    print("Normalize and flatten data...")
    print("x_fruits shape : " + str(x_fruits.shape))
    print(len(x_fruits))
    res, res2 = my_pca(x_fruits[:1000])
    plt.scatter(res[:,0],res[:,1])
    plt.show()
    res, res2 = my_pca(x_fruits[:10000])
    plt.scatter(res[:,0],res[:,1])
    plt.show()
    res, res2 = my_pca(x_fruits)
    plt.scatter(res[:,0],res[:,1])
    plt.show()


def test_pca_kmeans_fruit_dataset():
    x_fruits = np.load("test_images28.npy")
    y_fruits = np.load("test_labels28.npy")
    x_fruits = x_fruits.astype('float32') / 255.
    x_fruits = x_fruits.reshape((len(x_fruits), np.prod(x_fruits.shape[1:])))
    print("Normalize and flatten data...")
    print("x_fruits shape : " + str(x_fruits.shape))
    print(len(x_fruits))
    # res, res2 = my_pca(x_fruits[:1000])
    # lloyd_algorithm(res, 103, True)
    # res, res2 = my_pca(x_fruits[:10000])
    # lloyd_algorithm(res, 103, True)
    res, res2 = my_pca(x_fruits)
    lloyd_algorithm(res,y_fruits, 103, True)

def test_pca_mnist_2():
    import numpy as np, matplotlib.pyplot as plt
    from keras.datasets import mnist
    # Loading data from Mnist
    print("Loading data from Mnist...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalise data
    x_train = x_train.astype('float32') / 255.
    x_train_min = []
    y_train_min = []
    for i in range(60000):
        if y_train[i] == 0 or y_train[i] == 1:
            x_train_min.append(x_train[i])
            y_train_min.append(y_train[i])
    dataset = np.array(x_train_min)
    res, res2 = my_pca(dataset)
    plt.scatter(res[:,0],res[:,1], c=y_train_min)
    plt.show()

# all_test()
test_pca_kmeans_fruit_dataset()