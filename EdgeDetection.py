import numpy as np
import cv2
import h5py
from math import sqrt


def sobel(A, th):
    # A = image
    # th = threshold of intensities
    Gx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    Gy = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    rows = np.size(A, 0)
    columns = np.size(A, 1)
    mag = np.zeros((rows, columns))

    for row in range(rows - 2):
        for col in range(columns - 2):
            S1 = np.sum(np.sum(np.multiply(Gx, A[row:row+3, col:col+3])))
            S2 = np.sum(np.sum(np.multiply(Gy, A[row:row+3, col:col+3])))
            mag[row+1, col+1] = sqrt(S1**2 + S2**2)

    # Select threshold from 0 to 255 for intensity
    threshold = np.zeros((rows, columns)) + th
    output_image = np.maximum(mag, threshold)
    np.where(output_image, output_image < threshold, 0)
    return output_image