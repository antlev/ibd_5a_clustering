import keras
from keras.callbacks import *
from keras.layers import *
from keras.metrics import *
from keras.models import *
from keras.optimizers import *
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

experiment_name = "Mnist_Conv2D_auto_Encoders_test"

train_data_dir = '/home/soat/ibd_5a_clustering/fruits-360/test_images28.npy'
nb_train_samples = 17845
batch_size = 256

(x_train, y_train), (x_test, y_test) = np.load(train_data_dir)

print(x_train.shape)
exit(0)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
#
x_train = np.reshape(x_train, (-1, 28, 28, 1)) / 255.0
x_test = np.reshape(x_test, (-1, 28, 28, 1)) / 255.0
print(x_train.shape, y_train.shape)

tb_callback = TensorBoard("./logs/" + experiment_name)

##########################################################

input = Input(shape=(28, 28, 1))
encoder = Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1))(input)
encoder = MaxPooling2D((2, 2), padding='same')(encoder)
encoder = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)
encoder = MaxPooling2D((2, 2), padding='same')(encoder)
encoder = Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1))(encoder)
encoder = MaxPooling2D((2, 2), padding='same')(encoder)
encoder = Dense(2, activation='softmax')(encoder)
decoder = Flatten()(encoder)
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

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(28, 28),
    shuffle=True, seed=None,
    class_mode='categorical',
    interpolation='nearest')

autoencoder.compile(Adam(), mse, metrics=[categorical_accuracy])

autoencoder.fit_generator(train_generator, train_generator,
                          steps_per_epoch=nb_train_samples // batch_size,
                          epochs=1,
                          verbose=1,
                          validation_data=None,
                          validation_steps=None,
                          class_weight=None,
                          max_queue_size=10,
                          workers=1,
                          use_multiprocessing=False,
                          shuffle=True,
                          initial_epoch=0,
                          callbacks=[tb_callback])

encoded_imgs = encoder.predict(x_test)
predicted = autoencoder.predict(x_test)

plt.figure(figsize=(40, 4))
for i in range(10):
    # display original images
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display encoded images
    ax = plt.subplot(3, 20, i + 1 + 20)
    plt.imshow(encoded_imgs[i].reshape(8, 4))
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
