import numpy as np
import cv2
import matplotlib.pyplot as plt

def segment_image(image):
    img = cv2.imread(image)
    mask = np.zeros(img.shape[:2], 'uint8')

    bgModel = np.zeros((1,65), np.float64)
    fgModel = np.zeros((1,65), np.float64)

    h, w = img.shape[:2]
    rect = (10, 10, w - 10, h - 10)

    cv2.grabCut(img, mask, rect, bgModel, fgModel, 10, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented = img * mask2[:,:, np.newaxis]
    return segmented

