import os
import cv2
import numpy as np


path = "C:/Users/alevy/esgi/ml/ibd_5a_clustering/datasets/fruits-360_dataset/fruits-360/"

os.chdir(path)
directory = os.listdir('Test')

liste_image, liste_labels = [], []

for idx, classe in enumerate(directory):
    tmp = os.listdir('Test/' + classe)
    for i in tmp:
        img = cv2.imread('Test/' + classe + "/" + i)
        blank_image = cv2.resize(img,(28, 28))
        liste_image.append(img)
        liste_labels.append(idx)

x = np.asarray(liste_image)
y = np.asarray(liste_labels)

np.save("test_images28.npy", x)
np.save("test_labels28.npy", y)