
from scipy.stats import itemfreq
from skimage.feature import local_binary_pattern
import numpy as np
import cv2

def lbpfeature(img, eps=1e-7):
    if len(img.shape)==3:
        Ig=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        Ig=img.copy()
    
    radius = 3
    numPoints = 8 * radius
    numBins = 8*(8-1)+3

    lbp = local_binary_pattern(np.resize(Ig,[32,32]), numBins, radius, method="uniform")

    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0,numBins+1), range=(0, numBins + 2))

    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist
