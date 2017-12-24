from keras.models import Sequential, Model
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

if __name__ == "__main__":
    batch_size = 20
    tf.reset_default_graph()
    tensorboard = TensorBoard(log_dir="/output")
    epoch = 10
    X = []
    for filename in os.listdir('TrainSet/RGB/'):
        X.append(img_to_array(load_img('TrainSet/RGB/' + filename)))
    X = np.array(X, dtype=float)
    Xtrain = 1.0 / 255 * X

    inception = InceptionResNetV2(weights=None, include_top=True)
    inception.load_weights('LEARN_DATA/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
    inception.graph = tf.get_default_graph()
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

    # Fusion
    fusion_output = RepeatVector(32 * 32)(embed_input)
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([encoder_output, fusion_output], axis=3)
    fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output)

    # Decoder
    decoder_output = Conv2D(128, (3, 3), activation='relu', padding='same')(fusion_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(16, (3, 3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    COLOR_NET = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)
    COLOR_NET.compile(optimizer='adam', loss='mse')

    datagen = ImageDataGenerator(
        shear_range=0.4,
        zoom_range=0.4,
        rotation_range=40,
        horizontal_flip=True)
    COLOR_NET.fit_generator(image_a_b_gen(batch_size, datagen), callbacks=[tensorboard], epochs=epoch, steps_per_epoch=20)
    COLOR_NET.save('LEARN_DATA/My_Net.h5')
    model_json = COLOR_NET.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    COLOR_NET.save_weights("LEARN_DATA/color_tensorflow_real_mode.h5")

    Paint_img = []
    for filename in os.listdir('TrainSet/Grey/'):
        Paint_img.append(img_to_array(load_img('TrainSet/Grey/' + filename)))
    Paint_img = np.array(Paint_img, dtype=float)
    Paint_img = 1.0 / 255 * Paint_img
    Paint_img = gray2rgb(rgb2gray(Paint_img))
    Paint_img_embed = create_inception_embedding(Paint_img)
    Paint_img = rgb2lab(Paint_img)[:, :, :, 0]
    Paint_img = Paint_img.reshape(Paint_img.shape + (1,))

    output = COLOR_NET.predict([Paint_img, Paint_img_embed])
    output = output * 128

    for i in range(len(output)):
        cur = np.zeros((256, 256, 3))
        cur[:,:,0] = Paint_img[i][:, :, 0]
        cur[:,:,1:] = output[i]
        imsave("TrainSet/Result/"+str(i)+".png", lab2rgb(cur))



