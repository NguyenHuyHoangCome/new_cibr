import os
import numpy as np
from tensorflow.keras.preprocessing import image
from my_tools.autoencoder.autoencoder import train_model
from tensorflow.keras.models import Model, load_model


def get_features_img(pathdir):

    main = []
    for path in os.listdir(pathdir):
        img = image.load_img(pathdir+path, target_size=(256, 256))
        img = image.img_to_array(img)
        img_arr = np.expand_dims(img, axis=-1)
        main.append(img_arr)
    return main


def generate_features_autoencoder(train):

    print("Loading model...")
    model = load_model("../model/autoencoder.h5")

    # Creating the Encoder model of the Autoencoder
    encoder = Model(inputs=model.input, outputs=model.get_layer("encoder").output)

    # Generating the respective feature of images in latent vector spaces
    print("Creating the feature vectors...")
    features = encoder.predict(train)

    return features

if __name__ == '__main__':

    img_data = get_features_img('../data/')
    val_split = 0.2
    train_ds, val_ds = img_data[int(val_split * len(img_data)):], img_data[:int(val_split * len(img_data))]
    print("There are {} training images and {} images for validation".format(len(train_ds), len(val_ds)))

    train_ds, val_ds = np.array(train_ds), np.array(val_ds)
    train_ds, val_ds = train_ds.astype('float32') / 255.0, val_ds.astype('float32') / 255.0
    train_ds, val_ds = train_ds.squeeze(), val_ds.squeeze()

    print("Shape of train_ds : {}, val_ds : {}".format(train_ds.shape, val_ds.shape))

    train_model(train_ds, val_ds)

    print(generate_features_autoencoder(train_ds))