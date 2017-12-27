import os, sys
from skimage.io import imsave, imread
from skimage.transform import resize

def make_gray_img(data_path):
    Images = []
    path = 'TrainSet/RGB/'

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

    for filename in os.listdir(path):
        Images.append(imread(path + filename, as_grey=True))

    j = len(Images)
    if not os.path.isdir('TrainSet/Grey/'):
        os.makedirs('TrainSet/Grey/', exist_ok=True)
    for i in range(len(Images)):
        imsave('TrainSet/Grey/'+str(i), Images[i])
        progress = ((i * 100)//j) + 1
        sys.stdout.write("\rПрогресс: {} %   ".format(str(progress)))
    print("\nDone")

def make_trainset(data_path):
    Images = []
    path = 'TrainSet/RGB/'

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

    for filename in os.listdir(path):
        img = imread(path + filename)
        if img.shape < (128, 128, 3):
            print("Изображение {} имеет слишком маллый размер, из выборки удален".format(filename))
            os.remove(path+filename)
            continue
        Images.append(img)

    j = len(Images)

    for i in range(len(Images)):
        if Images[i].shape == (256, 256, 3):
            continue
        img = resize(Images[i], (256, 256, 3))
        imsave(path + str(i), img)
        progress = ((i * 100) // j) + 1
        sys.stdout.write("\rПрогресс: {} %   ".format(str(progress)))
    print("\nDone")