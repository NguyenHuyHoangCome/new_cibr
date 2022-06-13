import cv2
import numpy as np
def get_gcm_81(imgColor):
    img_hsv = cv2.cvtColor(imgColor,cv2.COLOR_BGR2HSV)

    w= img_hsv.shape[1]
    h= img_hsv.shape[0]
    list_81 = []
    for ch in range(3):
        Ig= img_hsv[:,:,ch]
        for k in range(3):
            for l in range(3):
                It = Ig[(k * h) // 3:((k + 1) * h) // 3, (l * w) // 3:((l + 1) * w) // 3]
                m, v, s = It.mean(), It.var(), It.std()
                list_81.append(m)
                list_81.append(v)
                list_81.append(s)
    return np.array(list_81)
