import os, sys
from skimage.io import imsave, imread

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

if __name__ == '__main__':
    make_gray_img(None)
