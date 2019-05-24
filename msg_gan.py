import time, os, random
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.python.keras import Input, Model
# from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.layers import Dense, LeakyReLU, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import TensorBoard

def named_logs(model, logs):
  result = {}
  for l in zip(model.metrics_names, [logs]):
    result[l[0]] = l[1]
  return result

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
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# generator, discriminator, generator_and_discriminator = my_gan(x_train,
#                                                                latent_space_dim=2,
#                                                                n_iterations_on_disc=5,
#                                                                iterations_max=100000,
#                                                                batch_size=512,
#                                                                save_res=True,
#                                                                save_iter=1000)

# generate_grid(generator, 10, 2)
# load_and_show(gan_folder="gan_1558610494")


def create_msg_model(data_shape_1, data_shape_2, data_shape_3, latent_space_dim ):

    input_1 = Input(shape=(784,))
    input_2 = Input(shape=(64,))
    input_3 = Input(shape=(16,))

    disc_1  = Dense(512)
    disc_2  = Dense(256)
    disc_3  = Dense(128)
    disc_4  = Dense(64)
    disc_5  = Dense(16)
    disc_6  = Dense(1)

    disc_tensor_output = \
        disc_6(disc_5(Concatenate()([input_3,
            disc_4(Concatenate()([input_2,
                disc_3(disc_2(disc_1(input_1)))
                                ]))])))
    discriminator_model = Model([input_1, input_2, input_3], disc_tensor_output)
    discriminator_model.compile(optimizer='adam', loss='mse')


    # plot_model(discriminator_model)
    latent_space_input = Input(shape=(latent_space_dim,))
    gen_1  = Dense(2)
    gen_2  = Dense(16)
    gen_3  = Dense(8)
    gen_4  = Dense(16)
    gen_5  = Dense(32)
    gen_6  = Dense(32)
    gen_7  = Dense(64)
    gen_8  = Dense(128)
    gen_9  = Dense(256)
    gen_10  = Dense(512)
    gen_11  = Dense(784)

    gen_1_output = gen_4(gen_3(gen_2(gen_1(latent_space_input))))
    gen_2_output = gen_7(gen_6(gen_5(gen_4(gen_3(gen_2(gen_1(latent_space_input)))))))
    gen_3_output = gen_11(gen_10(gen_9(gen_8(gen_7(gen_6(gen_5(gen_4(gen_3(gen_2(gen_1(latent_space_input)))))))))))
    gen_model = Model(latent_space_input, [gen_1_output, gen_2_output, gen_3_output] )
    gen_model.compile(optimizer='adam', loss='mse')

    disc_1.trainable = False
    disc_2.trainable = False
    disc_3.trainable = False
    disc_4.trainable = False
    disc_5.trainable = False
    disc_6.trainable = False

    print(gen_1_output.shape)
    print(gen_2_output.shape)
    print(gen_3_output.shape)

    gen_disc_output = \
        disc_6(disc_5(Concatenate()([gen_1_output,
            disc_4(Concatenate()([gen_2_output,
                disc_3(disc_2(disc_1(gen_3_output)))
                                ]))])))

    gen_disc = Model(latent_space_input, gen_disc_output)
    gen_disc.compile(optimizer='adam', loss='mse')

    return gen_disc, gen_model, discriminator_model

def my_msg_gan(data, data_2, data_3, n_iterations_on_disc=2, iterations_max=1000000, latent_space_dim=32, batch_size=256, save_res=False, save_iter=1000):
    seconds = int(time.time())

    nb_data = data.shape[0]
    data_shape_1 = data.shape[1]
    data_shape_2 = data_2.shape[1]
    data_shape_3 = data_3.shape[1]
    discriminator_tensorboard = TensorBoard(
      log_dir='./logs/' + str(seconds) + '/discriminator',
      histogram_freq=0,
      batch_size=batch_size,
      write_graph=True,
      write_grads=True
    )
    gen_disc_tensorboard = TensorBoard(
      log_dir='./logs/' + str(seconds) + '/gen_disc',
      histogram_freq=0,
      batch_size=batch_size,
      write_graph=True,
      write_grads=True
    )

    generator_and_discriminator_1, generator_1_model, discriminator_1_model, generator_and_discriminator_2, generator_2_model, discriminator_2_model, generator_and_discriminator_3, generator_3_model, discriminator_3_model = create_msg_model(data_shape_1, data_shape_2, data_shape_3, latent_space_dim)
    discriminator_tensorboard.set_model(discriminator_model)
    gen_disc_tensorboard.set_model(generator_and_discriminator)

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
        real_data = np.zeros((int(batch_size / 2), data_shape_1))
        for i in range(int(batch_size / 2)):
            real_data[i] = data[random.randint(0, nb_data - 1)]
        #     real_data[i] = np.random.choice((data, int(batch_size/2)))

        # Generate half of the batch using generator
        rand_gen = np.random.random((int(batch_size / 2), latent_space_dim))
        generated = generator_model.predict(rand_gen)
        # Associates targets
        real_data = np.asarray([[real_data[i], 1] for i in range((int(batch_size / 2)))])
        generated = np.asarray([[generated[i], 0] for i in range((int(batch_size / 2)))])
        # Concatenate, shuffle and traning on generator
        batch_to_train = np.concatenate((real_data, generated))
        np.random.shuffle(batch_to_train)
        batch_to_train_x = np.asarray([batch_to_train[i][0] for i in range(batch_size)])
        batch_to_train_y = np.asarray([batch_to_train[i][1] for i in range(batch_size)])
        for i in range(n_iterations_on_disc):
            discriminator_logs = discriminator_model.train_on_batch(batch_to_train_x, batch_to_train_y)
            if i == 0:
                discriminator_tensorboard.on_epoch_end(iterations, named_logs(discriminator_model, discriminator_logs))
        ### Generator Learning
        # Generate noise in latent_space shape and train the whole network
        gen_disc_logs = generator_and_discriminator.train_on_batch(np.random.random((batch_size, latent_space_dim)), np.full(batch_size, 1))
        gen_disc_tensorboard.on_epoch_end(iterations, named_logs(generator_and_discriminator, gen_disc_logs))

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

x_train_32 = np.resize(x_train, (x_train.shape[0], 32))
x_train_64 = np.resize(x_train, (x_train.shape[0], 64))



generator, discriminator, generator_and_discriminator = my_msg_gan(x_train,
                                                               x_train_64, 
                                                               x_train_32,
                                                               latent_space_dim=2,
                                                               n_iterations_on_disc=5,
                                                               iterations_max=100000,
                                                               batch_size=512,
                                                               save_res=True,
                                                               save_iter=1000)

# generate_grid(generator, 10, 2)
# load_and_show(gan_folder="gan_1558610494"))