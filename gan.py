import time, os, random
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)

def my_gan(data, n_iterations_on_disc=2, iterations_max=1000000, latent_space_dim=32, batch_size=2000, save_res=False, save_iter=1000):
    seconds = time.time()
    nb_data = data.shape[0]
    data_shape = data.shape[1]

    if save_res:
        folder = str("logs/gan_" + str(int(time.time())) + "/")
        logs_info = folder + "info"
        os.mkdir(folder)
        file = open(logs_info, "w")
        file.write("iterations_max : " + str(iterations_max) + " - batch_size : " + str(
            batch_size) + " - latent_space_dim : " + str(latent_space_dim) + " - nb_data : " + str(nb_data) + "\n")
        file.close()

    input = Input(shape=(784,))
    decoder = Dense(data_shape, activation='sigmoid')(input)
    decoder = Dense(64, activation='relu')(decoder)
    decoder = Dense(128, activation='relu')(decoder)
    decoder = Dense(1, activation='sigmoid')(decoder)

    discriminator = Model(input, decoder)
    discriminator.compile(optimizer='adam', loss='mse')

    decoder.trainable = False

    latent_space_input = Input(shape=(latent_space_dim,))
    encoder = Dense(64, activation='relu')(latent_space_input)
    encoder = Dense(128, activation='relu')(encoder)
    encoder = Dense(784, activation='sigmoid')(encoder)

    generator = Model(latent_space_input, encoder)
    generator.compile(optimizer='adam', loss='mse')

    encoded_data = generator(latent_space_input)
    output = discriminator(encoded_data)

    generator_and_discriminator = Model(latent_space_input, output)
    generator_and_discriminator.compile(optimizer='adam', loss='mse')

    iterations = 0

    while iterations < iterations_max:
        print("iteration " + str(iterations))
        ### Discriminator Learning
        # Take half of the batch in data
        real_data = np.zeros((int(batch_size / 2), data_shape))
        for i in range(int(batch_size / 2)):
            real_data[i] = data[random.randint(0, nb_data - 1)]
        #     real_data[i] = np.random.choice((data, int(batch_size/2)))

        # Generate half of the batch using generator
        rand_gen = np.random.random((int(batch_size / 2), latent_space_dim))
        generated = generator.predict(rand_gen)
        # Associates targets
        ###.
        real_data = np.asarray([[real_data[i], 1] for i in range((int(batch_size / 2)))])
        generated = np.asarray([[generated[i], 0] for i in range((int(batch_size / 2)))])
        ###
        # Concatenate, shuffle and traning on generator
        batch_to_train = np.concatenate((real_data, generated))
        np.random.shuffle(batch_to_train)
        batch_to_train_x = np.asarray([batch_to_train[i][0] for i in range(batch_size)])
        batch_to_train_y = np.asarray([batch_to_train[i][1] for i in range(batch_size)])
        for i in range(n_iterations_on_disc):
            discriminator.train_on_batch(batch_to_train_x, batch_to_train_y)
        ### Generator Learning
        # Generate noise in latent_space shape and train the whole network
        generator_and_discriminator.train_on_batch(np.random.random((batch_size, latent_space_dim)), np.full(batch_size, 1))
        if save_res and iterations%save_iter == 0:
            filename=str(int(time.time()))
            np.save(folder + filename, generator.predict(np.random.rand(1, latent_space_dim)))
            file = open(logs_info, "a+")
            file.write("file : " + filename + " - Iteration : %d\n" % iterations)
            file.close()
        iterations += 1
    print("Time to finish : " + str(int(time.time()-seconds)) + " sec batch_size = " + str(batch_size) + " (iterations:" + str(iterations_max) + ")")
    return generator, discriminator, generator_and_discriminator

def load_and_show(gan_folder="gan_1558556204"):
    directory="logs/" + gan_folder + "/"
    from os import listdir
    from os.path import isfile, join
    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    print("found " + str(len(files)) + " files !")
    for i in range(len(files)):
        if files[i] != "info":
            plt.imshow(np.load(directory + files[i]).reshape(28, 28))
            plt.gray()
            plt.show()


latent_space_dim=42
generator, discriminator, generator_and_discriminator = my_gan(x_train,latent_space_dim=latent_space_dim, iterations_max=1000000, save_res=True, save_iter=1000)

load_and_show()

