from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pylab as plt
from skimage import io
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d

def paint_border(img, patch, stride):
    img_h = img.shape[0]
    img_w = img.shape[1]
    leftover_h = (img_h - patch) % stride
    leftover_w = (img_w - patch) % stride
    print(leftover_h, leftover_w)
    if (leftover_h != 0):
        tmp_img = np.zeros((img_h + (stride - leftover_h), img_w))
        tmp_img[0:img_h, 0:img_w] = img
        img = tmp_img
    if (leftover_w != 0):  # change dimension of img_w
        tmp_img = np.zeros((img.shape[0], img.shape[1] + (stride - leftover_w)))
        tmp_img[0:img.shape[0], 0:img.shape[1]] = img
        img = tmp_img
    return img, img.shape[0], img.shape[1]

def patches(img, patch, stride):
    img_h = img.shape[0]
    img_w = img.shape[1]
    N_patches_tot = ((img_h - patch) // stride + 1) * ((img_w - patch) // stride + 1)
    patches = np.empty((N_patches_tot, patch, patch))
    iter_tot = 0
    for h in range((img_h - patch) // stride + 1):
        for w in range((img_w - patch) // stride + 1):
            patchh = img[int(h * stride):int((h * stride) + patch), int(w * stride):int((w * stride) + patch)]
            patches[iter_tot] = patchh
            iter_tot += 1  # total
    assert (iter_tot == N_patches_tot)
    return patches

def pred_to_imgs(pred, patch):
    pred_images = np.empty((pred.shape[0],pred.shape[1], pred.shape[2]))
    for i in range(pred.shape[0]):
        for h in range(pred.shape[1]):
            for w in range(pred.shape[2]):
                pred_images[i,h,w]=pred[i,h,w,0]
    #pred_images = np.reshape(pred_images,(pred_images.shape[0],1, patch, patch))
    return pred_images

def recompone_overlap(preds, img_h, img_w, stride):
    patch_h = preds.shape[1]
    patch_w = preds.shape[2]
    N_patches_h = (img_h-patch_h)//stride+1
    N_patches_w = (img_w-patch_w)//stride+1
    N_patches_img = N_patches_h * N_patches_w
    full_prob = np.zeros((img_h,img_w))
    full_sum = np.zeros((img_h,img_w))

    k = 0 #iterator over all the patches
    for h in range((img_h-patch_h)//stride+1):
        for w in range((img_w-patch_w)//stride+1):
            full_prob[h*stride:(h*stride)+patch_h,w*stride:(w*stride)+patch_w]+=preds[k]
            full_sum[h*stride:(h*stride)+patch_h,w*stride:(w*stride)+patch_w]+=1
            k+=1
    final_avg = full_prob/full_sum
    return final_avg

def to_keras_img(img):
    last = []
    for i in range(len(img)):
        tmp = []
        for j in range(len(img)):
            tmp.append([img[i][j]])
        last.append(tmp.copy())
    return last


if __name__ == "__main__":
    stride = 1
    img_test = (io.imread("data/maly.png", as_gray=True)/255)
    #image.array_to_img(to_keras_img(img_test)).show()
    border, new_h, new_w = paint_border(img_test, 48, stride)
    patcho = patches(border,48,stride)
    model = keras.models.load_model('model/model.h5')
    print("predicting")
    odp = model.predict(patcho, batch_size=1)
    print("predicted")
    pred_patches = pred_to_imgs(odp, 48)
    result = recompone_overlap(pred_patches, new_h, new_w,stride)
    image.array_to_img(to_keras_img(result)).show()
    '''
    img_test = image.img_to_array(image.load_img("data/maly.png", color_mode="grayscale"))/255
    cos = extract_patches_2d(img_test, (48, 48))
    more = []
    for x in range(len(cos)):
        last = []
        for i in range(len(cos[0])):
            tmp = []
            for j in range(len(cos[0])):
                tmp.append([cos[0][i][j]])
            last.append(tmp.copy())
        more.append(last.copy())
    ooo = reconstruct_from_patches_2d(np.array([cos[0],cos[1],cos[2],cos[3]]), (48, 48))
    last = []
    for i in range(len(ooo[0])):
        tmp = []
        for j in range(len(ooo[0])):
            tmp.append([ooo[i][j]])
        last.append(tmp.copy())
    image.array_to_img(last).show()
    image.array_to_img(more[1]).show()
    image.array_to_img(more[2]).show()
    image.array_to_img(more[3]).show()
    '''
    '''model = keras.models.load_model('model/model.h5')
    odp = model.predict(np.array([img_test]), batch_size=1)
    image.array_to_img(odp[0]).show()'''


