from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import random
import tensorflow as tf


if __name__ == "__main__":
    tf.reset_default_graph()
    epoch = 1000
    name = 'tiger'
    image = img_to_array(load_img(name+'.jpg'))
    image = np.array(image, dtype=float)
    X = rgb2lab(1.0/255*image)[:,:,0]
    Y = rgb2lab(1.0/255*image)[:,:,1:]
    Y /= 128
    X = X.reshape(1, 400, 400, 1)
    Y = Y.reshape(1, 400, 400, 2)

    COLORE_NET = Sequential()
    COLORE_NET.add(InputLayer(input_shape=(None, None, 1)))
    COLORE_NET.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    COLORE_NET.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    COLORE_NET.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    COLORE_NET.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    COLORE_NET.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    COLORE_NET.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    COLORE_NET.add(UpSampling2D((2, 2)))
    COLORE_NET.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    COLORE_NET.add(UpSampling2D((2, 2)))
    COLORE_NET.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    COLORE_NET.add(UpSampling2D((2, 2)))
    COLORE_NET.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))

    COLORE_NET.compile(optimizer='rmsprop', loss='mse')
    COLORE_NET.fit(x=X, y=Y, batch_size=1, epochs=epoch)
    print(COLORE_NET.evaluate(X, Y, batch_size=1))

    output = COLORE_NET.predict(X)
    output *= 128

    cur = np.zeros((400, 400, 3))
    cur[:,:,0] = X[0][:,:,0]
    cur[:,:,1:] = output[0]
    imsave(name+"_epochs"+str(epoch)+"_img_result.png", lab2rgb(cur))
    imsave(name+"_img_gray_version.png", rgb2gray(lab2rgb(cur)))

    COLORE_NET.save('/LEARN_DATA/')




