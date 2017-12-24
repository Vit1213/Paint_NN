import os, sys
from skimage.io import imsave, imread

def make_trainset():
    Images = []
    name_of_img = []
    for filename in os.listdir('TestSet/RGB/'):
        Images.append(imread('TrainSet/RGB/' + filename, as_grey=True))
        name_of_img.append(filename)

    j = len(Images)
    for i in range(len(Images)):
        imsave('TrainSet/Grey/'+name_of_img[i], Images[i])
        sys.stdout.write("\rProgress: {} %   ".format(str((i * 100)//j)))
    print("\nDone")

if __name__ == '__main__':
    make_trainset()
