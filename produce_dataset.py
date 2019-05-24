import os

import cv2
import numpy as np

path = "/home/soat/ibd_5a_clustering/fruits-360/"

os.chdir(path)
directory = os.listdir('Test')

list_image, list_labels = [], []

for idx, classe in enumerate(directory):
    tmp = os.listdir('Test/' + classe)
    for i in tmp:
        img = cv2.imread('Test/' + classe + "/" + i)
        image = cv2.resize(img, (28, 28))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        list_image.append(image)
        list_labels.append(idx)

x = np.asarray(list_image)
y = np.asarray(list_labels)

np.save("test_images28.npy", x)
np.save("test_labels28.npy", y)
