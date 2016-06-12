import numpy as np
from scipy.stats import norm
from skimage.restoration import denoise_tv_chambolle
from skimage.exposure import equalize_hist, equalize_adapthist
from scipy import ndimage
from scipy.misc import imresize
from keras.utils.generic_utils import Progbar


def crps(true, pred):
    """
    Calculation of CRPS.
    :param true: true values (labels)
    :param pred: predicted values
    """
    return np.sum(np.square(true - pred)) / true.size


def real_to_cdf(y, sigma=1e-10):
    """
    Utility function for creating CDF from real number and sigma (uncertainty measure).
    :param y: array of real values
    :param sigma: uncertainty measure. The higher sigma, the more imprecise the prediction is, and vice versa.
    Default value for sigma is 1e-10 to produce step function if needed.
    """
    cdf = np.zeros((y.shape[0], 600))
    for i in range(y.shape[0]):
        cdf[i] = norm.cdf(np.linspace(0, 599, 600), y[i], sigma)
    return cdf

def split_data_new(X, y, split_ratio=0.2):
    """
    Split data into training and testing.
    :param X: X
    :param y: y
    :param split_ratio: split ratio for train and test data
    """
    split = X.shape[0] * split_ratio
    X_test = X[:split, :, :, :]
    y_test = y[:split]
    X_train = X[split:, :, :, :]
    y_train = y[split:]

    return X_train, y_train, X_test, y_test
