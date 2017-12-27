from keras.models import load_model, Model
from keras.applications.inception_resnet_v2 import preprocess_input, InceptionResNetV2
from keras.layers import Conv2D, UpSampling2D, Input, Reshape, concatenate
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.io import imsave
from keras.callbacks import TensorBoard
from keras.layers.core import RepeatVector
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
from skimage.transform import resize
import os
import tensorflow as tf
import Color_text

Xtrain = []
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

def image_a_b_gen(batch_size, datagen):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        embed = create_inception_embedding(grayscaled_rgb)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield ([X_batch, create_inception_embedding(grayscaled_rgb)], Y_batch)

def make_model():
    embed_input = Input(shape=(1000,))

    encoder_input = Input(shape=(256, 256, 1,))
    encoder_output = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2)(encoder_input)
    encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
    encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same', strides=2)(encoder_output)
    encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)

    fusion_output = RepeatVector(32 * 32)(embed_input)
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([encoder_output, fusion_output], axis=3)
    fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output)

    decoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(fusion_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    return Model(inputs=[encoder_input, embed_input], outputs=decoder_output)

def begin_learn(epochs, batch_size, learn_path):
    global Xtrain
    global inception
    ERROR = Color_text.get_ERROR_color_text()
    WARNING = Color_text.get_WARNING_color_text()
    OK = Color_text.get_OK_color_text()
    batch_size = batch_size
    tf.reset_default_graph()
    path = 'TrainSet/RGB/'

    if learn_path:
        if os.path.isdir(learn_path):
            if len(os.listdir(learn_path)) != 0:
                path = learn_path
            else:
                print(WARNING+"Папка {} пуста, используется путь по умолчанию".format(learn_path))
        else:
            print(WARNING+"Путь {} не является директорией, используется путь по умолчанию".format(learn_path))

    if not os.path.isdir(path):
        print(ERROR+"Стандартный путь не доступен! Но мы его создали, заполните папку {} красивыми картиночками".format(path))
        os.makedirs(path, exist_ok=True)
        exit(1)
    elif os.listdir(path) == 0:
        print(ERROR+"Папка {} пуста".format(path))
        exit(1)

    tensorboard = TensorBoard(log_dir="/output")
    epoch = epochs
    X = []
    for filename in os.listdir(path):
        X.append(img_to_array(load_img(path + filename)))
    X = np.array(X, dtype=float)
    Xtrain = 1.0 / 255 * X

    inception = InceptionResNetV2(weights=None, include_top=True)
    if not os.access('LEARN_DATA/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5', os.R_OK):
        print(ERROR+"Файл весов для нейросети-классификатора не доступен")
        exit(1)
    inception.load_weights('LEARN_DATA/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
    inception.graph = tf.get_default_graph()
    print('Сборка модели...')
    if not os.access("LEARN_DATA/My_Net.h5", os.R_OK):
        COLOR_NET = make_model()
    else:
        COLOR_NET = load_model("LEARN_DATA/My_Net.h5")
    COLOR_NET.compile(optimizer='adam', loss='mse')

    datagen = ImageDataGenerator(
        shear_range=0.4,
        zoom_range=0.4,
        rotation_range=40,
        horizontal_flip=True)
    print('Запуск процедуры обучения...\n')
    COLOR_NET.fit_generator(image_a_b_gen(batch_size, datagen), callbacks=[tensorboard], epochs=epoch)
    print(OK)
    print("Сохраняю...")
    COLOR_NET.save('LEARN_DATA/My_Net.h5')
    model_json = COLOR_NET.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    COLOR_NET.save_weights("LEARN_DATA/color_tensorflow_real_mode.h5")
    print(OK)



