import random
import numpy as np
from numpy.linalg import eigh
from sklearn.decomposition import PCA

def covariance_matrix(data):
    centered = (data - np.mean(data.T,axis=1).T)
    return np.matmul(centered.T, centered) / (data.shape[0] - 1)

# def covariance(data):
#     nb_var = data.shape[1] # Assuming all data has the same nb of cols
#     nb_input = data.shape[0]
#     covariance = np.zeros([nb_var,nb_var])
#     moy_var1 = 0
#     moy_var2 = 0
#     for var_1 in range(nb_var):
#         print(str(var_1))
#         for var_2 in range(nb_var):
#             print(str(var_2))
#             for i in range(nb_input):
#                 moy_var1 += data[i][var_1]
#                 moy_var2 += data[i][var_2]
#             moy_var1 /= nb_input
#             moy_var2 /= nb_input
#             for i in range(nb_input):
#                 covariance[var_1,var_2] += (data[i][var_1] - moy_var1) * (data[i][var_2] - moy_var2)
#             covariance[var_1,var_2] /= nb_input-1
#             moy_var1 = 0
#             moy_var2 = 0
#     return covariance

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
    plt.scatter(res[:,0],res[:,1])
    plt.show()

def test_pca_mnist_2():
    import numpy as np, matplotlib.pyplot as plt
    from keras.datasets import mnist
    # Loading data from Mnist
    print("Loading data from Mnist...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalise data
    x_train = x_train.astype('float32') / 255.
    x_train_min = []
    for i in range(60000):
        if y_train[i] == 0 or y_train[i] == 1:
            x_train_min.append(x_train[i])
    dataset = np.array(x_train_min)
    res, res2 = my_pca(dataset)
    plt.scatter(res[:,0],res[:,1])
    plt.show()

all_test()
