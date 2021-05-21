from tensorflow.keras.preprocessing import image
import random
import numpy as np
import cv2
from skimage import io



def to_keras_img(img):
    last = []
    for i in range(len(img)):
        tmp = []
        for j in range(len(img)):
            tmp.append([img[i][j]])
        last.append(tmp.copy())
    return last

class ImageSampler():

    def __init__(self, name):
        #self.image = image.img_to_array(image.load_img("data/healthy/"+name+".jpg", color_mode="grayscale"))
        self.image = io.imread("data/healthy/"+name+".jpg", as_gray=True)
        image.array_to_img(to_keras_img(self.image)).show()
        self.mask = image.img_to_array(image.load_img("data/healthy_manualsegm/"+name+".tif", color_mode="grayscale"))/255
        self.fov = image.img_to_array(image.load_img("data/healthy_fovmask/"+name+"_mask.tif", color_mode="grayscale"))
        self.width = [len(self.image), len(self.image[0])]

    def imagee(self):
        return np.copy(self.image)

    def normalize(self, std, mean):
        self.image = (self.image-mean)/std
        self.image = ((self.image - np.min(self.image)) / (np.max(self.image) - np.min(self.image))) * 255

    def clahe(self):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.image = clahe.apply(self.image.astype(np.uint8))

    def gamma(self):
        invGamma = 1.0 / 1.2
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        self.image = cv2.LUT(np.array(self.image, dtype = np.uint8), table)

    def toFloat(self):
        self.image = self.image/255
        self.image = to_keras_img(self.image)
        image.array_to_img(self.image).show()


    def slice(self, n):
        counter = n
        points = []
        imgs = []
        masks = []
        while counter != 0:
            y = random.randint(0, self.width[1] - 49)
            x = random.randint(0, self.width[0] - 49)
            if self.fov[x, y] != 0 and [x, y] not in points:
                img = []
                mask = []
                for i in range(0, 48):
                    tmp = []
                    tmpm = []
                    for j in range(0, 48):
                        tmp.append(self.image[x+i][y+j])
                        tmpm.append(self.mask[x+i][y+j])
                    img.append(tmp.copy())
                    mask.append(tmpm.copy())
                imgs.append(img.copy())
                masks.append(mask.copy())
                points.append([x,y])
                counter -= 1
        return imgs, masks

class ImagesSampler():

    def __init__(self, names):
        self.imagess = []
        self.images = []
        images = []
        self.masks = []
        for i in names:
            a = ImageSampler(i)
            self.imagess.append(a)
            images.append(a.imagee())
        std = np.std(images)
        mean = np.mean(images)
        for i in self.imagess:
            i.normalize(std, mean)
            i.clahe()
            i.gamma()
            i.toFloat()

    def slices(self, n):
        for i in self.imagess:
            tmp = i.slice(n)
            self.images += tmp[0]
            self.masks += tmp[1]
        return np.array(self.images), np.array(self.masks)






if __name__ == "__main__":
    a = ImagesSampler(["01_h"])
    a.slices(3)