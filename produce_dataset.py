import os
import cv2
import numpy as np


path = "/Users/jean-luc/Documents/CoursESGI/5ibd/Clusteringavance/Projet"

#r = 100.0 / image.shape[1]
#dim = (100, int(image.shape[0] * r))

os.chdir(path)
directory = os.listdir('fruits-360/Test')

liste_image, liste_labels = [], []

os.chdir('fruits-360/Test')
for idx, classe in enumerate(directory):
    tmp = os.listdir(classe)
    for i in tmp:
        img = cv2.imread(tmp)
        try:
            img = cv2.resize(img, (64, 64))
            liste_image.append(img)
            liste_labels.append(idx)
        except:
            pass

X = np.asarray(liste_image)
y = np.asarray(liste_labels)

np.save("images64.npy")
np.save("labels64.npy")