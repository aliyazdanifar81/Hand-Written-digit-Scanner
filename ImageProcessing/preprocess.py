import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess(path: str):
    img = plt.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    img = cv2.ximgproc.niBlackThreshold(img, 255, cv2.THRESH_BINARY_INV, 41, -0.1,
                                        binarizationMethod=cv2.ximgproc.BINARIZATION_NICK)
    ker = np.ones((3, 3), dtype='uint8')
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, ker)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, ker)
    # img = cv2.morphologyEx(img, cv2.MORPH_ERODE, ker)
    # img = cv2.ximgproc.thinning(img.astype('uint8'))
    return img


def resizing(imges):
    result = []
    for i in range(len(imges)):
        result.append(cv2.resize(imges[i][0], (28, 28), interpolation=cv2.INTER_AREA))
    return result


def padding(img):
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value=0)
    return img