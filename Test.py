from keras.models import load_model
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.io import imsave
from keras.applications.inception_resnet_v2 import preprocess_input, InceptionResNetV2
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
from skimage.transform import resize
import tensorflow as tf

inception = object

def create_inception_embedding(grayscaled_rgb):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
    return embed


def begin_paint_img(data_path):
    global inception
    path = "TestSet/input/"

    if data_path:
        if os.path.isdir(data_path):
            if len(os.listdir(data_path)) != 0:
                path = data_path
            else:
                print("Папка {} пуста, используется путь по умолчанию".format(data_path))
                path = 'TrainSet/RGB/'
        else:
            print("Путь {} не является директорией, используется путь по умолчанию".format(data_path))

    if not os.path.isdir(path):
        print("Стандартный путь не доступен! Но мы его создали, заполните папку {} красивыми картиночками".format(path))
        os.makedirs(path, exist_ok=True)
        exit(1)

    if not os.access("LEARN_DATA/My_Net.h5", os.R_OK) or not os.access('LEARN_DATA/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5', os.R_OK):
        print("Нет файлов моделей!")
        exit(1)
    COLOR_NET = load_model("LEARN_DATA/My_Net.h5")
    inception = InceptionResNetV2(weights=None, include_top=True)
    inception.load_weights('LEARN_DATA/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
    inception.graph = tf.get_default_graph()

    Test_img = []
    for filename in os.listdir(path):
        Test_img.append(img_to_array(load_img(path + filename)))
    Test_img = np.array(Test_img, dtype=float)
    Test_img = 1.0 / 255 * Test_img
    Test_img = gray2rgb(rgb2gray(Test_img))
    Paint_img_embed = create_inception_embedding(Test_img)
    Test_img = rgb2lab(Test_img)[:, :, :, 0]
    Test_img = Test_img.reshape(Test_img.shape + (1,))

    output = COLOR_NET.predict([Test_img, Paint_img_embed])
    output = output * 128

    if not os.path.isdir('TestSet/Result/'):
        os.makedirs('TestSet/Result/', exist_ok=True)

    for i in range(len(output)):
        cur = np.zeros((256, 256, 3))
        cur[:, :, 0] = Test_img[i][:, :, 0]
        cur[:, :, 1:] = output[i]
        imsave("TestSet/Result/" + str(i) + ".png", lab2rgb(cur))
