# Applying Linear filter on image
# g(i,j) = K * f(i,j) + L
# where g(i,j) is pixel intensity of output at i,j and
# f(i,j) is pixel intensity of gray-scale image at i,j
# K and L are constants

import cv2
import numpy as np
import matplotlib.pyplot as plt


def linear_filter(image, K, L):
    '''
    Apply Linear filter to gray-scale image
    '''
    image = np.asarray(image, dtype=np.float)
    image = K * image + L
    # Clipping image
    image[image > 255] = 255
    image[image < 0] = 0
    image = np.asarray(image, dtype=np.int)
    return image


def main():
    # Read an image
    img = cv2.imread('../Images/sunflower.jpg')

    # Gray-scale of an image
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Linear filter
    K, L = 0.5, 0
    img_out1 = linear_filter(gray_img, K, L)

    K, L = 0.7, 20
    img_out2 = linear_filter(gray_img, K, L)

    K, L = 1.0, 10
    img_out3 = linear_filter(gray_img, K, L)

    outputs = np.hstack([gray_img, img_out1, img_out2, img_out3])
    plt.imshow(outputs, cmap='gray')
    plt.show()


if __name__==main():
    main()