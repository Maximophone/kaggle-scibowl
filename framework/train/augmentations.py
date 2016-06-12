import numpy as np
from scipy.stats import norm
from skimage.restoration import denoise_tv_chambolle
from skimage.exposure import equalize_hist, equalize_adapthist
from scipy import ndimage
from scipy.misc import imresize
from keras.utils.generic_utils import Progbar

def crop_k(img,k):
    assert img.shape[0] == img.shape[1]
    assert k<=1. and k>0, 'k must be between 0 and 1'
    x = img.shape[0]
    new_x = int(k*x)
    padding = (x-new_x)/2
    crop_img = img[padding:x-padding,padding:x-padding]
    
    return crop_img

def zoom(img,k):
    crop_img = crop_k(img,k)
    
    zoomed_img = imresize(crop_img.astype(float),img.shape)
    return zoomed_img

def sampling_augmentation(X, n):
    progbar = Progbar(X.shape[0])

    X_sampled = []
    for i in range(len(X)):
        slices = np.copy(X[i])
        ix = np.random.choice(range(len(slices)),n,replace=False)
        np.random.shuffle(ix)
        X_sampled.append(slices[ix,]) 
        progbar.add(1)
    return np.array(X_sampled)

def rotation(X, angle_range):
    progbar = Progbar(X.shape[0])  # progress bar for augmentation status tracking

    X_rot = np.copy(X)
    for i in range(len(X)):
        angle = np.random.randint(-angle_range, angle_range)
        for j in range(X.shape[1]):
            X_rot[i, j] = ndimage.rotate(X[i, j], angle, reshape=False, order=1)
        progbar.add(1)
    return X_rot


def shift(X, h_range, w_range):
    progbar = Progbar(X.shape[0])  # progress bar for augmentation status tracking

    X_shift = np.copy(X)
    size = X.shape[2:]
    for i in range(len(X)):
        h_random = np.random.rand() * h_range * 2. - h_range
        w_random = np.random.rand() * w_range * 2. - w_range
        h_shift = int(h_random * size[0])
        w_shift = int(w_random * size[1])
        for j in range(X.shape[1]):
            X_shift[i, j] = ndimage.shift(X[i, j], (h_shift, w_shift), order=0)
        progbar.add(1)
    return X_shift

def zoom_augmentation(X,y, k_min):
    progbar = Progbar(X.shape[0])  # progress bar for augmentation status tracking
    X_zoom = np.copy(X)
    y_zoom = np.copy(y)
    for i in range(len(X)):
        k_random = 1. - (np.random.rand() * (1. - k_min))
        for j in range(X.shape[1]):
            X_zoom[i, j] = zoom(X[i, j], k_random)
        y_zoom[i]*=1/(k_random*k_random)
        progbar.add(1)
    return X_zoom, y_zoom