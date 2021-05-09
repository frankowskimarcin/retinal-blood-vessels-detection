import cv2
from skimage import io, color
from skimage.filters import frangi
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import erosion
from sklearn.metrics import confusion_matrix


def resizeInputImage(dataDir, fileName):
    image = Image.open(dataDir + fileName)
    image = image.resize((1280, 853))
    image.save(dataDir + fileName[:-4] + "_small" + fileName[-4:])


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


def calculateEffectiveness(image, expertImage):
    image = image.reshape(-1)
    expertImage = expertImage.reshape(-1)
    cm = confusion_matrix(expertImage, image)

    # true positive, false positive, false negative, true negative
    tp, fp, fn, tn = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    print("TP: {}, FP: {}, FN: {}, TN:{}".format(tp, fp, fn, tn))

    acc = (tp + tn) / (tp + tn + fp + fn)               # accuracy
    sens = tp / (tp + fn)                               # sensitivity
    spec = tn / (tn + fp) if tn + fp != 0 else 0    # specificity
    print("Accuracy: {}, Sensitivity: {}, Specificity: {}".format(acc, sens, spec))


def main():
    dataDir = "data/healthy/"
    fileName = "02_h.jpg"
    fovMaskDir = "data/healthy_fovmask/"
    fovMaskFile = fileName[:-4] + "_mask.tif"
    expertMaskDir = "data/healthy_manualsegm/"
    expertMaskFile = fileName[:-4] + ".tif"
    resize = True

    if resize:
        resizeInputImage(dataDir, fileName)
        resizeInputImage(fovMaskDir, fovMaskFile)
        resizeInputImage(expertMaskDir, expertMaskFile)
        fileName = fileName[:-4] + "_small" + fileName[-4:]
        fovMaskFile = fovMaskFile[:-4] + "_small" + fovMaskFile[-4:]
        expertMaskFile = expertMaskFile[:-4] + "_small" + expertMaskFile[-4:]
        print(fileName)

    image = io.imread(dataDir + fileName)
    image = image[:, :, 1]
    fovImage = plt.imread(fovMaskDir + fovMaskFile)
    fovImage = fovImage[:, :, 1]
    expertImage = plt.imread(expertMaskDir + expertMaskFile)
    expertImage = expertImage.astype(np.uint8)
    # expertImage = expertImage[:, :, 1]
    # expertImage = cv2.cvtColor(expertImage, cv2.COLOR_GRAY2BGR)
    # expertImage = cv2.cvtColor(expertImage, cv2.COLOR_BGR2GRAY)

    frangiImage = frangiFilter(image, fovImage)
    frangiImage = erosion(frangiImage)
    frangiImage = frangiImage.astype(np.uint8)

    calculateEffectiveness(frangiImage, expertImage)

    # io.imshow(frangiImage)
    # io.show()
    # io.imsave('data/output/out.jpg', frangiImage)


if __name__ == '__main__':
    main()
