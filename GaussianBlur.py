# Computer Vision Project: Ignorant Vision
# Gaussian Blur Function
# Created with source for guidance:
# http://www.adeveloperdiary.com/data-science/computer-vision/applying-gaussian-smoothing-to-an-image-using-python-from-scratch/


import cv2
import numpy as np
import matplotlib.pyplot as plt


def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def convolution(image, kernel, average=False, verbose=False):
    if len(image.shape) == 3:
        # print("Found 3 Channels : {}".format(image.shape))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # print("Converted to Gray Channel. Size : {}".format(image.shape))
    # else:
        # print("Image Shape : {}".format(image.shape))

    # print("Kernel Shape : {}".format(kernel.shape))

    if verbose:
        plt.imshow(image, cmap='gray')
        plt.title("Image")
        plt.show()

    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    # padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
    # padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    padded_image = cv2.copyMakeBorder(image, pad_height, pad_width, pad_height, pad_width, cv2.BORDER_REPLICATE) 
    
    if verbose:
        plt.imshow(padded_image, cmap='gray')
        plt.title("Padded Image")
        plt.show()

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
            if average:
                output[row, col] /= kernel.shape[0] * kernel.shape[1]

    # print("Output Image size : {}".format(output.shape))

    if verbose:
        plt.imshow(output, cmap='gray')
        plt.title("Output Image using {}X{} Kernel".format(kernel_row, kernel_col))
        plt.show()
        cv2.imwrite("outputP6.png", output)

    return output


def gaussian_kernel(size, sigma=1, verbose=False):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    kernel_2D *= 1.0 / kernel_2D.max()

    if verbose:
        plt.imshow(kernel_2D, interpolation='none', cmap='gray')
        plt.title("Kernel ( {}X{} )".format(size, size))
        plt.show()

    return kernel_2D


def gaussian_blur(image, kernel_size, verbose=False):
    kernel = gaussian_kernel(kernel_size, sigma=np.sqrt(2), verbose=verbose)
    return convolution(image, kernel, average=True, verbose=verbose)


if __name__ == '__main__':
    image = cv2.imread("inputP6.jpg", cv2.IMREAD_GRAYSCALE)

    gaussian_blur(image, 9, verbose=True)
