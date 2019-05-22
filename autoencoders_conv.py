import keras
from PIL import Image
from keras.callbacks import *
from keras.datasets import *
from keras.layers import *
from keras.models import *
from matplotlib import pyplot as plt

experiment_name = "Mnist_Conv2D_auto_Encoders_adam/adam_10E"

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

x_train = np.reshape(x_train, (-1, 28, 28, 1)) / 255.0
x_test = np.reshape(x_test, (-1, 28, 28, 1)) / 255.0

tb_callback = TensorBoard("./logs/" + experiment_name)

##########################################################

input = Input(shape=(28, 28, 1))
encoder = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1))(input)
encoder = MaxPooling2D((2, 2), padding='same')(encoder)
encoder = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)
encoder = MaxPooling2D((2, 2), padding='same')(encoder)
encoder = Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1))(encoder)
encoder = MaxPooling2D((2, 2), padding='same')(encoder)
encoder = Flatten()(encoder)
encoder = Dense(2, activation='relu')(encoder)
decoder = Dense(392, activation='relu')(encoder)
decoder = Reshape((7, 7, 8))(decoder)
decoder = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(decoder)
decoder = UpSampling2D((2, 2))(decoder)
decoder = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(decoder)
decoder = UpSampling2D((2, 2))(decoder)
decoder = Conv2DTranspose(1, (3, 3), activation='relu', padding='same')(decoder)

###############################################

autoencoder = Model(input, decoder)
encoder = Model(input, encoder)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                batch_size=128,
                epochs=5,
                shuffle=True,
                callbacks=[tb_callback])

encoded_imgs = encoder.predict(x_test)
predicted = autoencoder.predict(x_test)
# decoder = decoder.pre

plt.figure(figsize=(40, 4))
for i in range(10):
    # display original images
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display encoded images
    # ax = plt.subplot(3, 20, i + 1 + 20)
    # print(encoded_imgs[i].shape)
    # plt.imshow(encoded_imgs[i].reshape(8, 4))
    # plt.gray()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

    # display reconstructed images
    ax = plt.subplot(3, 20, 2 * 20 + i + 1)
    plt.imshow(predicted[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# generated_inputs = np.asarray([[x, y] for x in np.arange(0, 1, 0.1)
#                                for y in np.arange(0, 1, 0.1)], dtype=np.float32)
# generated_imgs = predicted(generated_inputs, verbose=0)
# for img in generated_imgs[0::10]:
#     Image.fromarray(img * 255.0).show()
