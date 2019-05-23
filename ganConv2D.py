import os
import random
import time

import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, LeakyReLU, Conv2D, MaxPooling2D, Flatten, Conv2DTranspose, UpSampling2D, Reshape
from keras.models import Model
from matplotlib import pyplot as plt


def my_gan(data, n_iterations_on_disc=2, iterations_max=1000000, latent_space_dim=2, batch_size=256, save_res=False,
           save_iter=1000):
    seconds = time.time()
    nb_data = data.shape[0]
    data_shape = data.shape[1]

    if save_res:
        folder = str("logs/gan_" + str(int(time.time())) + "/")
        print(folder)
        logs_info = folder + "info"
        os.mkdir(folder)
        file = open(logs_info, "w")
        file.write("iterations_max : " + str(iterations_max) + " - batch_size : " + str(
            batch_size) + " - latent_space_dim : " + str(latent_space_dim) + " - nb_data : " + str(nb_data) + "\n")
        file.close()

    input = Input(shape=(28, 28, 1))
    discriminator = Conv2D(16, (3, 3), activation=LeakyReLU(0.3), padding='same')(input)
    discriminator = MaxPooling2D((2, 2), padding='same')(discriminator)
    discriminator = Conv2D(8, (3, 3), activation=LeakyReLU(0.3), padding='same')(discriminator)
    discriminator = MaxPooling2D((2, 2), padding='same')(discriminator)
    discriminator = Conv2D(8, (3, 3), activation=LeakyReLU(0.3), padding='same')(discriminator)
    discriminator = Flatten()(discriminator)

    discriminator = Dense(1, activation='sigmoid')(discriminator)

    discriminator_model = Model(input, discriminator)
    discriminator_model.compile(optimizer='adam', loss='mse')

    discriminator.trainable = False

    latent_space_input = Input(shape=(latent_space_dim,))

    generator = Dense(392, activation=LeakyReLU(0.3))(latent_space_input)
    generator = Reshape((7, 7, 8))(generator)
    generator = Conv2DTranspose(8, (3, 3), activation=LeakyReLU(0.3), padding='same')(generator)
    generator = UpSampling2D((2, 2))(generator)
    generator = Conv2DTranspose(8, (3, 3), activation=LeakyReLU(0.3), padding='same')(generator)
    generator = UpSampling2D((2, 2))(generator)
    generator = Conv2DTranspose(1, (3, 3), activation=LeakyReLU(0.3), padding='same')(generator)

    generator_model = Model(latent_space_input, generator)
    generator_model.compile(optimizer='adam', loss='mse')

    encoded_data = generator_model(latent_space_input)
    output = discriminator_model(encoded_data)

    generator_and_discriminator = Model(latent_space_input, output)
    generator_and_discriminator.compile(optimizer='adam', loss='mse')

    iterations = 0

    while iterations < iterations_max:
        print("iteration " + str(iterations))
        # Discriminator Learning
        # Take half of the batch in data
        real_data = np.zeros((int(batch_size / 2), data_shape))
        for i in range(int(batch_size / 2)):
            real_data[i] = data[random.randint(0, nb_data - 1)]
        #     real_data[i] = np.random.choice((data, int(batch_size/2)))

        # Generate half of the batch using generator
        rand_gen = np.random.random((int(batch_size / 2), latent_space_dim))
        generated = generator_model.predict(rand_gen)
        # Associates targets
        ###.
        real_data_labels = np.asarray([1 for i in range((int(batch_size / 2)))])
        generated_labels = np.asarray([0 for i in range((int(batch_size / 2)))])
        ###
        # Concatenate, shuffle and traning on generator
        batch_to_train_x = np.concatenate((real_data.reshape((-1, 28, 28, 1)), generated))
        batch_to_train_y = np.concatenate((real_data_labels, generated_labels))
        shuffled_indexes = np.arange((len(batch_to_train_x)))
        np.random.shuffle(shuffled_indexes)
        batch_to_train_x = batch_to_train_x[shuffled_indexes]
        batch_to_train_y = batch_to_train_y[shuffled_indexes]
        for i in range(n_iterations_on_disc):
            discriminator_model.train_on_batch(batch_to_train_x, batch_to_train_y)
        # Generator Learning
        # Generate noise in latent_space shape and train the whole network
        generator_and_discriminator.train_on_batch(np.random.random((batch_size, latent_space_dim)),
                                                   np.full(batch_size, 1))
        if save_res and iterations % save_iter == 0:
            filename = str(int(time.time()))
            np.save(folder + filename, generator_model.predict(np.random.rand(1, latent_space_dim)))
            file = open(logs_info, "a+")
            file.write("file : " + filename + " - Iteration : %d\n" % iterations)
            file.close()
        iterations += 1
    print("Time to finish : " + str(int(time.time() - seconds)) + " sec batch_size = " + str(
        batch_size) + " (iterations:" + str(iterations_max) + ")")
    return generator_model, discriminator_model, generator_and_discriminator


def load_and_show(gan_folder="gan_1558556204"):
    directory = "logs/" + gan_folder + "/"
    from os import listdir
    from os.path import isfile, join
    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    print("found " + str(len(files) - 1) + " files !")
    for i in range(len(files)):
        if files[i] != "info":
            plt.imshow(np.load(directory + files[i]).reshape(28, 28))
            plt.gray()
            plt.show()


def generate_grid(generator, grid_size, latent_space_dim=2):
    fig = plt.figure(figsize=(50, 50))
    columns = 10
    rows = 10
    for i in range(1, columns * rows + 1):
        img = generator.predict(np.array([[0.1 * i / columns, 0.1 * i % columns]])).reshape(28, 28)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        plt.gray()
    plt.show()


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

latent_space_dim = 20
# generator, discriminator, generator_and_discriminator = my_gan(x_train,
#                                                                latent_space_dim=latent_space_dim,
#                                                                n_iterations_on_disc=1,
#                                                                iterations_max=10000,
#                                                                batch_size=32,
#                                                                save_res=True, save_iter=1000)
# generate_grid(generator, 10, 2)
load_and_show(gan_folder="gan_1558621375")
