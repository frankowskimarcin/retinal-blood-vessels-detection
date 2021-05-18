import cv2
from skimage import io, color, filters, measure
from skimage.filters import frangi
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import erosion
from sklearn.metrics import confusion_matrix
from sklearn import neighbors
import random


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
    resize = False

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
    io.imsave('data/output/out.jpg', frangiImage)

class drugie():
    def __init__(self, fileName):
        self.dataDir = "data/CHASEDB1/"
        self.image = plt.imread(self.dataDir + fileName+".jpg")
        self.image = filters.gaussian(self.image, 3)
        self.center = [len(self.image)//2, len(self.image[0])//2]
        self.expert = plt.imread(self.dataDir + fileName+"_1stHO.png")
        self.classifier = neighbors.KNeighborsClassifier()

    def statCalc(self, point):
        frag = []
        for i in range(-2, 3):
            try:
                frag.append(self.image[point[0]+i][point[1]-2: point[1]+3])
            except:
                pass
        frag = np.array(frag)
        sdr = np.var(frag[:,:,0].flatten())
        sdg = np.var(frag[:,:,1].flatten())
        sdb = np.var(frag[:,:,2].flatten())
        gFrag = color.rgb2gray(frag)
        centr = measure.moments_central(np.array(gFrag))
        centrNorm = measure.moments_normalized(centr)
        hu = measure.moments_hu(centrNorm)
        result = []
        result.append(sdr)
        result.append(sdg)
        result.append(sdb)
        for i in hu:
            result.append(i)
        return result


    def run(self):
        points = []
        params = []
        labels = []
        num = 10000
        while num != 0:
            y = random.randint(0, self.center[0])
            x = random.randint(0, self.center[1])
            if [x, y] not in points and np.sum(self.image[x, y])!=0:
                points.append([x, y])
                params.append(self.statCalc([x, y]))
                labels.append(self.expert[x, y])
                num -= 1
        self.classifier.fit(params, labels)
        resImg = np.zeros((len(self.image), len(self.image[0])))
        for i in range(len(self.image)):
            for j in range(len(self.image[0])):
                if np.sum(self.image[i, j])!=0:
                    stat = self.statCalc([i, j])
                    resImg[i,j] = self.classifier.predict([stat])[0]
            print(i)
        plt.imshow(resImg)
        plt.show()





if __name__ == '__main__':
    a = drugie("Image_01L")
    a.run()
    #main()
