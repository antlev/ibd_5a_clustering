import random
import numpy as np

def covariance(data):
    nb_var = len(data) # Assuming all data has the same nb of cols
    nb_input = len(data[0])
    covariance = np.zeros([nb_var,nb_var])
    moy_var1 = 0
    moy_var2 = 0
    for var_1 in range(nb_var):
        for var_2 in range(nb_var):
            for i in range(nb_input):
                moy_var1 += data[var_1][i]
                moy_var2 += data[var_2][i]
            moy_var1 /= nb_input
            moy_var2 /= nb_input
            for i in range(nb_input):
                covariance[var_1,var_2] += (data[var_1][i] - moy_var1) * (data[var_2][i] - moy_var2)
            covariance[var_1,var_2] /= nb_input-1
            moy_var1 = 0
            moy_var2 = 0
    return covariance


test = np.array([[random.random(), random.random(), random.random()],[random.random(), random.random(), random.random()],[random.random(), random.random(), random.random()]])

print("Our result : ")
print(covariance(test))
print("numpy result : ")
print(np.cov(test))