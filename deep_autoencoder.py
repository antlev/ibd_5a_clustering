from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()

# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# print (x_train.shape)
# print (x_test.shape)

x_fruits = np.load("images64color.npy")
x_fruits = x_fruits.astype('float32') / 255.
x_fruits = x_fruits.reshape((len(x_fruits), np.prod(x_fruits.shape[1:])))
print("Normalize and flatten data...")
print("x_fruits shape : " + str(x_fruits.shape))

from keras.layers import Input, Dense, LeakyReLU
from keras.models import Model
from keras import regularizers

encoding_dim = 2
input_img = Input(shape=(64*64*3,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='relu')(decoded)
decoded = Dense(64*64*3, activation='relu')(decoded)

encoder = Model(input_img, encoded)
autoencoder=Model(input_img, decoded)
encoder = Model(input_img, encoded)
autoencoder=Model(input_img, decoded)

encoded_input = Input(shape=(encoding_dim,))
deco = autoencoder.layers[-4](encoded_input)
deco = autoencoder.layers[-3](deco)
deco = autoencoder.layers[-2](deco)
deco = autoencoder.layers[-1](deco)
decoder = Model(encoded_input, deco)

autoencoder.compile(optimizer='adam', loss='logcosh')

# autoencoder.fit(x_train, x_train,
#                 epochs=50,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_data=(x_test, x_test))

autoencoder.fit(x_fruits, x_fruits,
                epochs=1,
                batch_size=256,
                shuffle=True)


# encoded_imgs = encoder.predict(x_test)
# predicted = autoencoder.predict(x_test)
#
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(40, 4))
# for i in range(10):
#     # display original images
#     ax = plt.subplot(3, 20, i + 1)
#     plt.imshow(x_fruits[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # # display encoded images
#     # ax = plt.subplot(3, 20, i + 1 + 20)
#     # plt.imshow(encoded_imgs[i].reshape(2, 1))
#     # plt.gray()
#     # ax.get_xaxis().set_visible(False)
#     # ax.get_yaxis().set_visible(False)
#
#     # display reconstructed images
#     ax = plt.subplot(3, 20, 20 + i + 1)
#     plt.imshow(predicted[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

# plt.show()

import random
x=np.random.rand(100*100, 2)
for i in range(100):
  for j in range(100):
    x[j+100*i][0]=i*0.01
    x[j+100*i][1]=j*0.01
print(x)
generated = decoder.predict(x)

import matplotlib.pyplot as plt

plt.figure(figsize=(40, 4))
for i in range(10000):
    # display original images
    ax = plt.subplot(100, 100, i + 1)
    plt.imshow(generated[i].reshape(64, 64, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()