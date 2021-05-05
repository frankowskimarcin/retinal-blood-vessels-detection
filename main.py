import cv2
from skimage import io, color
from skimage.filters import frangi
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def resizeInputImage(dataDir, fileName, fovMaskDir, fovMaskFile):
    image = Image.open(dataDir + fileName)
    image = image.resize((1280, 853))
    image.save(dataDir + fileName[:-4] + "_small" + fileName[-4:])

    fovImage = Image.open(fovMaskDir + fovMaskFile)
    fovImage = fovImage.resize((1280, 853))
    fovImage.save(fovMaskDir + fovMaskFile[:-4] + "_small" + fovMaskFile[-4:])


def frangiFilter(image, fovImage):
    image = frangi(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if fovImage[y][x] == 0:
                image[y][x] = 0
            if image[y][x] > 0.0000002:
                image[y][x] = 255
            else:
                image[y][x] = 0

    return image


def main():
    dataDir = "data/healthy/"
    fileName = "02_h.jpg"
    fovMaskDir = "data/healthy_fovmask/"
    fovMaskFile = '02_h_mask.tif'
    resize = True

    if resize:
        resizeInputImage(dataDir, fileName, fovMaskDir, fovMaskFile)
        fileName = fileName[:-4] + "_small" + fileName[-4:]
        fovMaskFile = fovMaskFile[:-4] + "_small" + fovMaskFile[-4:]
        print(fileName)
        print(fovMaskFile)

    image = io.imread(dataDir + fileName)
    image = image[:, :, 1]
    fovImage = plt.imread(fovMaskDir + fovMaskFile)
    fovImage = fovImage[:, :, 1]

    frangiImage = frangiFilter(image, fovImage)

    io.imshow(frangiImage)
    io.show()
    io.imsave('data/output/out.jpg', frangiImage)


if __name__ == '__main__':
    main()
