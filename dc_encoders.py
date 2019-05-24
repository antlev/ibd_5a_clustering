from keras.callbacks import *
from keras.layers import *
from keras.metrics import *
from keras.models import *
from keras.optimizers import *
from matplotlib import pyplot as plt

experiment_name = "DCEncoders_loss:mse_20epoch"

train_data_dir = '/home/soat/ibd_5a_clustering/fruits-360/test_images28.npy'
nb_train_samples = 17845
batch_size = 256

# (x_train, y_train), (x_test, y_test) = np.load(train_data_dir)
x_train = np.load(train_data_dir)

print(x_train.shape)

x_train = np.reshape(x_train, (-1, 28, 28, 1)) / 255.0
print(x_train.shape)

tb_callback = TensorBoard("./logs/" + experiment_name)

##########################################################

input = Input(shape=(28, 28, 1))
encoder = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1))(input)
encoder = MaxPooling2D((2, 2), padding='same')(encoder)
encoder = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)
encoder = MaxPooling2D((2, 2), padding='same')(encoder)
encoder = Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1))(encoder)
encoder = MaxPooling2D((2, 2), padding='same')(encoder)
decoder = Flatten()(encoder)
encoder = Dense(2, activation='softmax')(encoder)
decoder = Dense(392, activation='softmax')(decoder)
decoder = Reshape((7, 7, 8))(decoder)
decoder = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(decoder)
decoder = UpSampling2D((2, 2))(decoder)
decoder = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(decoder)
decoder = UpSampling2D((2, 2))(decoder)
decoder = Conv2DTranspose(1, (3, 3), activation='relu', padding='same')(decoder)

###############################################

autoencoder = Model(input, decoder)
encoder = Model(input, encoder)

autoencoder.compile(Adam(), mse)

autoencoder.fit(x_train, x_train,
                batch_size=32,
                epochs=20,
                shuffle=True,
                callbacks=[tb_callback])

encoded_imgs = encoder.predict(x_train)
predicted = autoencoder.predict(x_train)

plt.figure(figsize=(40, 4))
for i in range(10):
    # display original images
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(x_train[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    # display reconstructed images
    ax = plt.subplot(3, 20, 2 * 20 + i + 1)
    plt.imshow(predicted[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
