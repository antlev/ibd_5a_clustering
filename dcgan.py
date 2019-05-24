import os
import random
import time

import numpy as np
from keras.layers import Input, Dense, LeakyReLU, Conv2D, MaxPooling2D, Flatten, Conv2DTranspose, UpSampling2D, Reshape
from keras.models import Model
from matplotlib import pyplot as plt
from tensorboard_logger import configure, log_value

configure("./logs/30000_iterations_mse_color", flush_secs=5)


# def wasserstein_loss(y_true, y_pred):
#     """Calculates the Wasserstein loss for a sample batch.
#     The Wasserstein loss function is very simple to calculate. In a standard GAN, the
#     discriminator has a sigmoid output, representing the probability that samples are
#     real or generated. In Wasserstein GANs, however, the output is linear with no
#     activation function! Instead of being constrained to [0, 1], the discriminator wants
#     to make the distance between its output for real and generated samples as
#     large as possible.
#     The most natural way to achieve this is to label generated samples -1 and real
#     samples 1, instead of the 0 and 1 used in normal GANs, so that multiplying the
#     outputs by the labels will give you the loss immediately.
#     Note that the nature of this loss means that it can be (and frequently will be)
#     less than 0."""
#     return K.mean(y_true * y_pred)


def my_gan(data, n_iterations_on_disc=2, iterations_max=1000000, latent_space_dim=2, batch_size=256, save_res=False,
           save_iter=1000):
    seconds = time.time()
    nb_data = data.shape[0]
    data_shape = data.shape[1:]

    if save_res:
        folder = str("logs/gan_" + str(int(time.time())) + "/")
        print(folder)
        logs_info = folder + "info"
        os.mkdir(folder)
        file = open(logs_info, "w")
        file.write("iterations_max : " + str(iterations_max) + " - batch_size : " + str(
            batch_size) + " - latent_space_dim : " + str(latent_space_dim) + " - nb_data : " + str(nb_data) + "\n")
        file.close()

    input = Input(shape=(28, 28, 3))
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
    generator = Conv2DTranspose(3, (3, 3), activation=LeakyReLU(0.3), padding='same')(generator)

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
        real_data = np.zeros(((int(batch_size / 2),) + data_shape))
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
        batch_to_train_x = np.concatenate((real_data.reshape((-1, 28, 28, 3)), generated))
        batch_to_train_y = np.concatenate((real_data_labels, generated_labels))
        shuffled_indexes = np.arange((len(batch_to_train_x)))
        np.random.shuffle(shuffled_indexes)
        batch_to_train_x = batch_to_train_x[shuffled_indexes]
        batch_to_train_y = batch_to_train_y[shuffled_indexes]
        for i in range(n_iterations_on_disc):
            loss = discriminator_model.train_on_batch(batch_to_train_x, batch_to_train_y)
            log_value('d_loss', loss, iterations)
        # Generator Learning
        # Generate noise in latent_space shape and train the whole network
        loss = generator_and_discriminator.train_on_batch(np.random.random((batch_size, latent_space_dim)),
                                                          np.full(batch_size, 1))
        log_value('g_loss', loss, iterations)
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
            plt.imshow(np.load(directory + files[i]).reshape(28, 28, 3))
            plt.show()


def generate_grid(generator, grid_size, latent_space_dim=2):
    fig = plt.figure(figsize=(50, 50))
    columns = 10
    rows = 10
    for i in range(1, columns * rows + 1):
        img = generator.predict(np.array([[0.1 * i / columns, 0.1 * i % columns]])).reshape(28, 28, 3)
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


train_data_dir = '/home/soat/ibd_5a_clustering/fruits-360/images28color.npy'
nb_train_samples = 17845
batch_size = 256
x_train = np.load(train_data_dir)

print(x_train.shape)

x_train = np.reshape(x_train, (-1, 28, 28, 3)) / 255.0
print(x_train.shape)

# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# print(x_train.shape)


latent_space_dim = 2
generator, discriminator, generator_and_discriminator = my_gan(x_train,
                                                               latent_space_dim=2,
                                                               n_iterations_on_disc=1,
                                                               iterations_max=20000,
                                                               batch_size=16,
                                                               save_res=True,
                                                               save_iter=2000)
generate_grid(generator, 10, 2)
# load_and_show(gan_folder="gan_1558710778")
