import cv2 as cv
from scipy.ndimage import binary_fill_holes
import numpy as np
from scipy.ndimage import binary_fill_holes

DEFAULT_KERNEL_SIZE = (5, 5)


def gamma_correction(frame, gamma):
    inv = 1. / gamma
    table = ((np.arange(256) / 255.) * inv) * 255.
    return cv.LUT(frame, table).astype(np.uint8)


def detect_balls(frame, detector, gamma=.5): # np.ndarray
    corrected = gamma_correction(frame, gamma=gamma)
    hsv_frame = cv.cvtColor(corrected, cv.COLOR_BGR2HSV)
    # XXX: We use the saturation channel because it extract the background easily
    # If there is a light issue use the follwing value for gray image
    # gray = hsv_frame[..., 0]
    gray = hsv_frame[..., 1]  # toggle between 0/1???
    _, th = cv.threshold(gray,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,DEFAULT_KERNEL_SIZE)
    opened = cv.morphologyEx(th, cv.MORPH_OPEN, kernel, iterations=2)
    filled = binary_fill_holes(~opened)
    filled = (~filled).astype(np.uint8) * 255
    keypoints = detector.detect(filled)
    return keypoints
