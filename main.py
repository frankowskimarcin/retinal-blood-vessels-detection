from skimage import io, color, filters, measure, transform
from skimage.filters import frangi
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import erosion
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import neighbors, tree, model_selection
import random
import multiprocessing
from imblearn.under_sampling import RandomUnderSampler
from imblearn.metrics import classification_report_imbalanced


# from tensorflow.keras import layers
# from tensorflow import keras
# from tensorflow.keras.preprocessing import image
# import tensorflow as tf
# import images


def calculateEffectiveness(image, expertImage):
    print(classification_report_imbalanced(image.flatten().astype(np.uint8) * 255, expertImage.flatten()))
    image = image.reshape(-1)
    expertImage = expertImage.reshape(-1)
    cm = confusion_matrix(expertImage, image)

    # true positive, false positive, false negative, true negative
    tp, fp, fn, tn = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    print("TP: {}, FP: {}, FN: {}, TN:{}".format(tp, fp, fn, tn))

    acc = (tp + tn) / (tp + tn + fp + fn)  # accuracy
    sens = tp / (tp + fn)  # sensitivity
    spec = tn / (tn + fp) if tn + fp != 0 else 0  # specificity
    print("Accuracy: {}, Sensitivity: {}, Specificity: {}".format(acc, sens, spec))


class First:

    def __init__(self, selected_image):
        self.dataDir = "data/healthy/"
        # fileName = "02_h.jpg"
        self.fovMaskDir = "data/healthy_fovmask/"
        self.expertMaskDir = "data/healthy_manualsegm/"
        self.resize = False
        self.fileName = selected_image
        self.fovMaskFile = self.fileName[:-4] + "_mask.tif"
        self.expertMaskFile = self.fileName[:-4] + ".tif"

    def resizeInputImage(self, dataDir, fileName):
        image = Image.open(dataDir + fileName)
        image = image.resize((1280, 853))
        image.save(dataDir + fileName[:-4] + "_small" + fileName[-4:])

    def frangiFilter(self, image, fovImage):
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

    def execute(self):
        # if self.resize:
        #     resizeInputImage(self.dataDir, self.fileName)
        #     resizeInputImage(self.fovMaskDir, self.fovMaskFile)
        #     resizeInputImage(self.expertMaskDir, self.expertMaskFile)
        #     fileName = self.fileName[:-4] + "_small" + self.fileName[-4:]
        #     fovMaskFile = self.fovMaskFile[:-4] + "_small" + self.fovMaskFile[-4:]
        #     expertMaskFile = self.expertMaskFile[:-4] + "_small" + self.expertMaskFile[-4:]
        #     print(fileName)

        image = io.imread(self.dataDir + self.fileName)
        image = image[:, :, 1]
        fovImage = plt.imread(self.fovMaskDir + self.fovMaskFile)
        fovImage = fovImage[:, :, 1]
        expertImage = plt.imread(self.expertMaskDir + self.expertMaskFile)
        expertImage = expertImage.astype(np.uint8)
        # expertImage = expertImage[:, :, 1]
        # expertImage = cv2.cvtColor(expertImage, cv2.COLOR_GRAY2BGR)
        # expertImage = cv2.cvtColor(expertImage, cv2.COLOR_BGR2GRAY)

        frangiImage = self.frangiFilter(image, fovImage)
        frangiImage = erosion(frangiImage)
        frangiImage = frangiImage.astype(np.uint8)
        calculateEffectiveness(frangiImage, expertImage)

        io.imshow(frangiImage)
        io.show()
        io.imsave('data/output/out.jpg', frangiImage)


class Second():
    def __init__(self, train):
        self.dataDir = "data/healthy/"
        self.images = []
        self.widths = []
        self.experts = []
        self.fovs = []
        for fileName in train:
            img = filters.gaussian(plt.imread(self.dataDir + fileName + ".jpg"))
            self.widths.append([len(img), len(img[0])])
            self.images.append(img)
            self.experts.append(plt.imread("data/healthy_manualsegm/" + fileName + ".tif").astype(np.uint8))
            self.fovs.append(color.rgb2gray(plt.imread("data/healthy_fovmask/" + fileName + "_mask" + ".tif")))
        self.classifier = neighbors.KNeighborsClassifier(n_neighbors=3)
        self.tree = tree.DecisionTreeClassifier()

    def statCalc(self, img, point):
        frag = []
        for i in range(-3, 4):
            try:
                frag.append(img[point[0] + i][point[1] - 2: point[1] + 3])
            except:
                pass
        frag = np.array(frag)
        # sdr = np.var(frag[:,:,0].flatten())
        sdg = np.var(frag[:, :, 1].flatten())
        # sdb = np.var(frag[:,:,2].flatten())
        gFrag = color.rgb2gray(frag).astype(np.uint8)
        result = []
        result.append(np.median(gFrag))
        # result.append(np.average(gFrag))
        result.append(np.var(gFrag))
        # result.append(sdr)
        result.append(sdg)
        # result.append(sdb)
        return result

    def train(self):
        params = []
        labels = []
        num = 7000
        counter = 0
        which = 0
        while num != 0:
            y = random.randint(0, self.widths[which][1] - 1)
            x = random.randint(0, self.widths[which][0] - 1)
            if self.fovs[which][x, y] != 0:
                params.append(self.statCalc(self.images[which], [x, y]))
                labels.append(self.experts[which][x, y])
                num -= 1
                counter += 1
            if counter >= 700:
                which += 1
                counter = 0
        self.classifier.fit(x, y)

    def run(self, imgname):
        print("start {}".format(imgname))
        image = filters.gaussian(plt.imread(self.dataDir + imgname + ".jpg"))
        fov = color.rgb2gray(plt.imread("data/healthy_fovmask/" + imgname + "_mask" + ".tif"))
        resImg = np.zeros((len(image), len(image[0]))).astype(np.uint8)
        for i in range(len(image)):
            for j in range(len(image[0])):
                if fov[i, j] != 0:
                    stat = self.statCalc(image, [i, j])
                    resImg[i, j] = self.classifier.predict([stat])[0]
            print(i)
        print("end {}".format(imgname))
        io.imsave('data/output/' + imgname + 'out.jpg', resImg)

    def trainTree(self):
        params = []
        labels = []
        num = 11000
        counter = 0
        which = 0
        while num != 0:
            y = random.randint(0, self.widths[which][1] - 1)
            x = random.randint(0, self.widths[which][0] - 1)
            if self.fovs[which][x, y] != 0:
                params.append(self.statCalc(self.images[which], [x, y]))
                labels.append(self.experts[which][x, y])
                num -= 1
                counter += 1
            if counter >= 1100:
                which += 1
                counter = 0
        rus = RandomUnderSampler(sampling_strategy=0.6)
        X, y = rus.fit_resample(params, labels)
        print(len(labels), len(y))
        self.tree.fit(X, y)

    def runTree(self, imgname):
        print("start {}".format(imgname))
        image = filters.gaussian(plt.imread(self.dataDir + imgname + ".jpg"))
        fov = color.rgb2gray(plt.imread("data/healthy_fovmask/" + imgname + "_mask" + ".tif"))
        resImg = np.zeros((len(image), len(image[0]))).astype(np.uint8)
        for i in range(len(image)):
            for j in range(len(image[0])):
                if fov[i, j] != 0:
                    stat = self.statCalc(image, [i, j])
                    resImg[i, j] = self.tree.predict([stat])[0]
            print(i)
        print("end {}".format(imgname))
        io.imsave('data/output/' + imgname + '_tree_out.png', resImg)

class SecondShow():
    def __init__(self, name):
        self.image = io.imread("data/healthy/"+name)
        self.mask = io.imread("data/healthy_manualsegm/"+name[0:name.find(".")]+".tif")
        self.result = io.imread("data/output/"+name[0:name.find(".")]+"_tree_out.jpg")
        for i in range(self.result.shape[0]):
            for j in range(self.result.shape[1]):
                if(self.result[i,j]>220):
                    self.result[i,j] = 255
                else:
                    self.result[i, j] = 0


    def show(self):
        io.imshow(self.result)
        io.show()
        io.imshow(self.image)
        io.show()
        calculateEffectiveness(self.result, self.mask)





def starting_process():
    print("starting {}".format(multiprocessing.current_process().name))


if __name__ == '__main__':
    jpegs = ["01_h", "02_h", "03_h", "04_h", "05_h", "06_h", "07_h",
             "08_h", "09_h", "10_h", "11_h", "12_h", "13_h", "14_h", "15_h"]
    '''a = drugie(jpegs[0:10])
    a.trainTree()
    #for i in jpegs:
    a.runTree(jpegs[0])'''
    # calculateEffectiveness(plt.imread("data/output/out.jpg"), plt.imread("data/healthy_manualsegm/02_h.tif"))
    '''pool_size = 6
    a = drugie(jpegs[0:10])
    a.train()
    a_pool = multiprocessing.Pool(processes=pool_size,
                                  initializer=starting_process)
    a_pool.map(a.run, jpegs[10:16])
    a_pool.close()
    a_pool.join()'''
    # for i in jpegs[10:16]:
    # a.run(i)
    # main()
    s = SecondShow("02_h.jpg")
    s.show()
    #f = First("02_h.jpg")
    #f.execute()
