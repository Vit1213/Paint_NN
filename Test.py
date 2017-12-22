from keras.models import Sequential, load_model
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import random
import tensorflow as tf

if __name__ == "__main__":
    name = 'flow'
    image = img_to_array(load_img(name + '.jpg'))
    image = np.array(image, dtype=float)
    X = rgb2lab(1.0 / 255 * image)[:, :, 0]
    X = X.reshape(1, 400, 400, 1)
    COLORE_NET = load_model('My_Net.h5')

    output = COLORE_NET.predict(X)
    output *= 128

    cur = np.zeros((400, 400, 3))
    cur[:, :, 0] = X[0][:, :, 0]
    cur[:, :, 1:] = output[0]
    imsave(name + "_TEST" + "_img_result.png", lab2rgb(cur))
    imsave(name + "_img_gray_version.png", rgb2gray(lab2rgb(cur)))
