import time, os, random
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense, LeakyReLU, BatchNormalization


def create_model(data_shape, latent_space_dim ):

    input = Input(shape=(784,))
    discriminator = Dense(data_shape)(input)
    discriminator  = LeakyReLU()(discriminator )
    discriminator = BatchNormalization()(discriminator)
    discriminator  = Dense(512)(discriminator )
    discriminator  = LeakyReLU()(discriminator )
    discriminator = BatchNormalization()(discriminator)
    discriminator  = Dense(256)(discriminator )
    discriminator  = LeakyReLU()(discriminator )
    discriminator = BatchNormalization()(discriminator)

    discriminator  = Dense(256)(discriminator )
    discriminator  = LeakyReLU()(discriminator )
    discriminator = BatchNormalization()(discriminator)

    discriminator  = Dense(128)(discriminator )
    discriminator  = LeakyReLU()(discriminator )
    discriminator = BatchNormalization()(discriminator)

    discriminator  = Dense(64)(discriminator )
    discriminator  = LeakyReLU()(discriminator )
    discriminator = BatchNormalization()(discriminator)

    discriminator  = Dense(1, activation='sigmoid')(discriminator )

    discriminator_model = Model(input, discriminator )
    discriminator_model.compile(optimizer='adam', loss='mse')

    discriminator .trainable = False

    latent_space_input = Input(shape=(latent_space_dim,))
    generator = Dense(64)(latent_space_input)
    generator = LeakyReLU()(generator)
    generator = BatchNormalization()(generator)

    generator = Dense(128)(generator)
    generator = LeakyReLU()(generator)
    generator = BatchNormalization()(generator)

    generator = Dense(256)(generator)
    generator = LeakyReLU()(generator)
    generator = BatchNormalization()(generator)

    generator = Dense(256)(generator)
    generator = LeakyReLU()(generator)
    generator = BatchNormalization()(generator)

    generator = Dense(512)(generator)
    generator = LeakyReLU()(generator)
    generator = BatchNormalization()(generator)

    generator = Dense(784)(generator)
    generator = LeakyReLU()(generator)


    generator_model = Model(latent_space_input, generator)
    generator_model.compile(optimizer='adam', loss='mse')

    encoded_data = generator_model(latent_space_input)
    output = discriminator_model(encoded_data)

    generator_and_discriminator = Model(latent_space_input, output)
    generator_and_discriminator.compile(optimizer='adam', loss='mse')

    return generator_and_discriminator, generator_model, discriminator_model

def my_gan(data, n_iterations_on_disc=2, iterations_max=1000000, latent_space_dim=32, batch_size=256, save_res=False, save_iter=1000):
    seconds = time.time()
    nb_data = data.shape[0]
    data_shape = data.shape[1]

    generator_and_discriminator, generator_model, discriminator_model = create_model(data_shape, latent_space_dim)

    iterations = 0
    if save_res:
        folder = str("logs/gan_" + str(int(time.time())) + "/")
        print("data generated from generator will be save in : " + folder)
        logs_info = folder + "info"
        os.mkdir(folder)
        file = open(logs_info, "w")
        file.write("iterations_max : " + str(iterations_max) + " - batch_size : " + str(
            batch_size) + " - latent_space_dim : " + str(latent_space_dim) + " - nb_data : " + str(nb_data) + "\n")
        file.write("GAN structure : \n")
        file.write(str(generator_and_discriminator.summary()) + "\n")
        file.close()

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
        generated = generator_model.predict(rand_gen)
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
            discriminator_model.train_on_batch(batch_to_train_x, batch_to_train_y)
        ### Generator Learning
        # Generate noise in latent_space shape and train the whole network
        generator_and_discriminator.train_on_batch(np.random.random((batch_size, latent_space_dim)), np.full(batch_size, 1))
        if save_res and iterations%save_iter == 0:
            generate_grid(generator_model, 10, latent_space_dim)
            filename=str(int(time.time()))
            np.save(folder + filename, generator_model.predict(np.random.rand(1, latent_space_dim)))
            file = open(logs_info, "a+")
            file.write("file : " + filename + " - Iteration : %d\n" % iterations)
            file.close()
        iterations += 1
    print("Time to finish : " + str(int(time.time()-seconds)) + " sec batch_size = " + str(batch_size) + " (iterations:" + str(iterations_max) + ")")
    if save_res:
        print("data generated from generator are saved in : " + folder)
    generate_grid(generator_model, 10, latent_space_dim)
    return generator_model, discriminator_model, generator_and_discriminator

def load_and_show(gan_folder="gan_1558556204"):
    directory="logs/" + gan_folder + "/"
    from os import listdir
    from os.path import isfile, join
    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    print("found " + str(len(files)-1) + " files !")
    for i in range(len(files)):
        if files[i] != "info":
            plt.imshow(np.load(directory + files[i]).reshape(28, 28))
            plt.gray()
            plt.show()

def generate_test_images(generator, nb_images, latent_space_dim=2):
    for i in range(nb_images):
        plt.imshow(generator.predict(np.random.rand(1, latent_space_dim)).reshape(28, 28))
        plt.gray()
        plt.show()

def generate_grid(generator, grid_size, latent_space_dim=2):
    fig = plt.figure(figsize=(50, 50))
    columns = 10
    rows = 10
    for i in range(1, columns * rows + 1):
        img = generator.predict(np.array([[0.1*i/columns, 0.1*i%columns]])).reshape(28, 28)
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

generator, discriminator, generator_and_discriminator = my_gan(x_train,
                                                               latent_space_dim=2,
                                                               n_iterations_on_disc=2,
                                                               iterations_max=1000000,
                                                               batch_size=32,
                                                               save_res=True,
                                                               save_iter=10000)

generate_grid(generator, 10, 2)
# load_and_show(gan_folder="gan_1558610494")

