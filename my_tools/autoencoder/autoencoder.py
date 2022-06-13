# Importing relevant libraries
import numpy as np
from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, Conv2DTranspose, Activation, Flatten, ReLU, Dense, Reshape
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt


# Create Model

def create_autoencoder_model(height, width, channels, latent_image_size=32):

    input_shape = (height, width, channels)

    ## Building Encoder Model
    # Feed the input to the model
    inputs = Input(shape=input_shape)

    # Feeding to Convolutional network and applying Batch Normalization (et al. Ioffe and Szegedy, 2015)
    x = Conv2D(filters=8, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(x)
    x = BatchNormalization(axis=-1)(x)

    # Generating the latent vectors after flattening
    enc_size = K.int_shape(x)
    x = Flatten()(x)
    encoder = Dense(latent_image_size, name='encoder')(x)

    ## Building Decoder Model

    # Feed the output of the Encoder to the Decoder model
    x = Dense(np.prod(enc_size[1:]))(encoder)
    x = Reshape((enc_size[1], enc_size[2], enc_size[3]))(x)

    # Feeding to Convolutional network and applying Batch Normalization
    # in reverse order as it is decoder. Conv2DTranspose applies Convolutional network
    # while Upsampling simultaneously
    x = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2DTranspose(filters=8, kernel_size=(3, 3), strides=2, activation='relu', padding='same')(x)
    x = BatchNormalization(axis=-1)(x)

    # Feeding it to another Conv2DTranspose to recover original Input shape
    x = Conv2DTranspose(filters=channels, kernel_size=(3, 3), padding='same')(x)
    decoder = Activation('sigmoid', name='decoder')(x)

    # Constructing the Autoencoder Model
    autoencoder = Model(inputs, decoder, name='autoencoder')

    return autoencoder

def train_model(train, val, epochs=18, lr=1e-3, batch_size=64):

    print("Creating model...")
    model = create_autoencoder_model(256, 256, 3)
    model.compile(optimizer=Adam(lr=lr), loss="mse")

    # Train the model
    history = model.fit(
        train, train,
        validation_data=(val, val),
        epochs=epochs,
        batch_size=batch_size)

    # Saving the model in HDF5 format
    model.save('model/autoencoder.h5', save_format="h5")

