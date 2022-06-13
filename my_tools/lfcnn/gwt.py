import cv2
import pylab as pl
import glob
import pandas as pd
import numpy as np


# define gabor filter bank with different orientations and at different scales
def build_filters():
    filters = []
    ksize = 9
    # define the range for theta and nu
    for theta in np.arange(0, np.pi, np.pi / 8):
        for scale in range(1, 5 + 1):
            kern = cv2.getGaborKernel((ksize, ksize), scale, theta, 10, 0.5, 0, ktype=cv2.CV_32F)
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    return filters


# function to convolve the image with the filters
def process(img, filters):
    v_gwt = []
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        v_gwt.append(np.mean(fimg))
        v_gwt.append(np.var(fimg))
        v_gwt.append(np.std(fimg))

        # intialize pca and logistic regression model
    return np.array(v_gwt)


def gwtfeature(img):
    filters = build_filters()
    f = np.asarray(filters)
    if len(img.shape)==3:
        Ig=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        Ig=img.copy()
    Ig=cv2.resize(Ig,(64,64))
    v_wgt=process(Ig,filters)
    return v_wgt


