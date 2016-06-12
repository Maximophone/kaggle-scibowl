import numpy as np
from scipy.stats import norm
from skimage.restoration import denoise_tv_chambolle
from skimage.exposure import equalize_hist, equalize_adapthist
from scipy import ndimage
from scipy.misc import imresize
from keras.utils.generic_utils import Progbar

def cut(image,th1,th2):
    meds = (image >= th1)&(image < th2)
    maxs = image >= th2
    final = np.ones(image.shape)*.5*meds + np.ones(image.shape)*maxs
    return final

def preprocess1(X,weight=0.1):
    """
    Pre-process images that are fed to neural network.
    :param X: X
    """
    progbar = Progbar(X.shape[0])  # progress bar for pre-processing status tracking

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] = denoise_tv_chambolle(X[i, j], weight=weight, multichannel=False)
        progbar.add(1)
    return X


def preprocess2(X,weight=0.1):
    """
    Pre-process images that are fed to neural network.
    :param X: X
    """
    progbar = Progbar(X.shape[0])  # progress bar for pre-processing status tracking

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] = denoise_tv_chambolle(X[i, j], weight=weight, multichannel=False)
            X[i, j] = equalize_hist(X[i, j])
            X[i, j] = cut(X[i, j],0.33,0.66)
        progbar.add(1)
    return X

def preprocess3(X,weight=0.1):
    """
    Pre-process images that are fed to neural network.
    :param X: X
    """
    progbar = Progbar(X.shape[0])  # progress bar for pre-processing status tracking

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] = denoise_tv_chambolle(X[i, j], weight=weight, multichannel=False)
            X[i, j] = equalize_adapthist(X[i, j])
            # X[i, j] = cut(X[i, j],0.33,0.66)
        progbar.add(1)
    return X